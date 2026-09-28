import base64
import io
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    text: str
    yes_probability: Optional[float] = None
    token_logprobs: Optional[List[Dict[str, Any]]] = None


class VLMBackend:
    def generate(self, image_path: Optional[str], prompt: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    def generate_result(
        self,
        image_path: Optional[str],
        prompt: str,
        max_new_tokens: int = 256,
        request_yes_probability: bool = False,
    ) -> GenerationResult:
        return GenerationResult(text=self.generate(image_path, prompt, max_new_tokens=max_new_tokens))


class TransformersQwenBackend(VLMBackend):
    def __init__(self, model_path: str, device_map: str = "auto", torch_dtype: str = "bfloat16"):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, torch_dtype)
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, image_path: Optional[str], prompt: str, max_new_tokens: int = 256) -> str:
        content = []
        if image_path:
            content.append({"type": "image", "image": str(Path(image_path).resolve())})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class OpenAICompatibleBackend(VLMBackend):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def generate(self, image_path: Optional[str], prompt: str, max_new_tokens: int = 256) -> str:
        return self.generate_result(image_path, prompt, max_new_tokens=max_new_tokens).text

    def generate_result(
        self,
        image_path: Optional[str],
        prompt: str,
        max_new_tokens: int = 256,
        request_yes_probability: bool = False,
    ) -> GenerationResult:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(image_path)},
                }
            )
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }
        if request_yes_probability:
            kwargs.update({"logprobs": True, "top_logprobs": 20})
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            if not request_yes_probability:
                raise
            kwargs.pop("logprobs", None)
            kwargs.pop("top_logprobs", None)
            response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        token_rows = self._extract_token_logprobs(getattr(choice, "logprobs", None))
        return GenerationResult(
            text=text,
            yes_probability=self._yes_probability(token_rows),
            token_logprobs=token_rows or None,
        )

    @staticmethod
    def _extract_token_logprobs(logprobs: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        for token_info in (content or [])[:6]:
            top = []
            for candidate in getattr(token_info, "top_logprobs", None) or []:
                top.append(
                    {
                        "token": str(getattr(candidate, "token", "")),
                        "logprob": float(getattr(candidate, "logprob", float("-inf"))),
                    }
                )
            rows.append(
                {
                    "token": str(getattr(token_info, "token", "")),
                    "logprob": float(getattr(token_info, "logprob", float("-inf"))),
                    "top_logprobs": top,
                }
            )
        return rows

    @staticmethod
    def _yes_probability(rows: List[Dict[str, Any]]) -> Optional[float]:
        for row in rows:
            candidates = list(row.get("top_logprobs", []))
            candidates.append({"token": row.get("token", ""), "logprob": row.get("logprob")})
            deduplicated = {}
            for candidate in candidates:
                key = str(candidate.get("token", "")).strip().lower()
                try:
                    logprob = float(candidate.get("logprob"))
                except (TypeError, ValueError):
                    continue
                deduplicated[key] = max(logprob, deduplicated.get(key, float("-inf")))
            yes_mass = 0.0
            no_mass = 0.0
            for raw_token, logprob in deduplicated.items():
                token = raw_token.strip(".,:;!?")
                try:
                    mass = math.exp(logprob)
                except (TypeError, ValueError, OverflowError):
                    continue
                if token in {"yes", "y", "true"}:
                    yes_mass += mass
                elif token in {"no", "n", "false"}:
                    no_mass += mass
            if yes_mass + no_mass > 0:
                return yes_mass / (yes_mass + no_mass)
        return None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        raw = path.read_bytes()
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                max_pixels = 4_000_000
                max_edge = 2400
                scale = min(
                    1.0,
                    (max_pixels / max(1, width * height)) ** 0.5,
                    max_edge / max(1, width, height),
                )
                if scale < 1.0:
                    resized = image.convert("RGB").resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    buffer = io.BytesIO()
                    resized.save(buffer, format="JPEG", quality=95, optimize=True)
                    raw = buffer.getvalue()
                    mime = "image/jpeg"
        except Exception:
            pass
        data = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{data}"


def build_backend(args) -> VLMBackend:
    if args.backend == "transformers":
        if not args.model_path:
            raise ValueError("--model-path is required for transformers backend.")
        return TransformersQwenBackend(
            model_path=args.model_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )
    if args.backend == "openai":
        return OpenAICompatibleBackend(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")
