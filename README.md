# VPC3：多类别视觉答案校准

VPC3 面向视觉语言模型已经给出的答案，按问题的原生答案类型调用专家生成候选，再由冻结门控决定保留原答案还是采用候选。五类答案保持原样：对象存在为 Yes/No，颜色为颜色词，空间关系题返回相关实体，计数返回整数，图中文字问答返回自由文本。

当前 VPC3 提供 Python、命令行和本机 HTTP 接口。入口可连接 Qwen3-VL 服务，也可接收 SAM3、OCR 等外部专家的观测；它不负责启动这些大模型服务。输出的证据是来源与观测记录，不等同于形式化证明。

## 工作流程

1. 用基座视觉语言模型对图像和问题生成原始答案。
2. 仅根据问题识别答案类型和需要的视觉证据。
3. 对应专家产生与原答案同类型的候选答案，并附上观测信息。
4. 冻结门控评估候选相对于原答案的增量效用；证据不足时保留原答案。
5. 输出请求 ID、原答案、校准答案、修改决定和所用证据；答案类型只在内部路由使用。

## 输入与输出

完整字段见 [VPC3 接口说明](docs/vpc_integration_api_v1.md)。单题入口接收请求 ID、图像、问题和原答案。下例通过 OpenAI 兼容服务调用 Qwen3-VL 专家：

```json
{
  "request_id": "example-001",
  "image": {"source": "path", "value": "/path/to/image.jpg"},
  "question": "How many red cups are on the table?",
  "original_answer": "2"
}
```

先使用 Python 3.10 安装代码依赖：

```bash
python -m pip install -r requirements.txt -r requirements-calibration.txt
```

```bash
python scripts/run_vpc3.py \
  --input request.json \
  --output response.json \
  --qwen3-base-url http://localhost:8000/v1
```

返回示例（仅说明格式，不代表示例图像的真实推理结果）：

```json
{
  "request_id": "example-001",
  "original_answer": "2",
  "calibrated_answer": "3",
  "changed": true,
  "evidence": {
    "source": "qwen3_vl_8b",
    "observation": "独立视觉语言模型的候选回答；未提供检测框或文字定位",
    "candidate_answer": "3",
    "reason": "candidate_passed_gate"
  }
}
```

需要 HTTP 集成时，可启动 `python scripts/serve_vpc3.py --qwen3-base-url http://localhost:8000/v1`，向 `http://127.0.0.1:8765/calibrate` 发送同一 JSON。服务默认只监听本机，`GET /health` 用于健康检查。图像路径由运行 VPC3 的机器读取；跨机器调用时可使用 `image.source=base64`。

`evidence` 说明候选来源、观测、候选答案和门控原因。专家分数、阈值和内部路由不进入默认输出。若对接方已有专家服务，可在输入中增加可选的 `expert_evidence`；其字段格式和支持范围见接口说明。未连接 Qwen3-VL 且未提供专家证据时，入口会报错。

## 对接入口

- `scripts/run_vpc3.py`：单题 JSON 命令行入口。
- `scripts/serve_vpc3.py`：本机 HTTP `POST /calibrate` 接口。
- `woodpecker_qwen/vpc3_integration.py`：可直接导入的 Python 校准器。
- `artifacts/vpc3_gate/`：随仓库提供的门控文件和 SHA256 校验清单。

## 对接部署所需工件

门控文件 `gate_model.json`、`gate_model.joblib` 和校验清单位于 `artifacts/vpc3_gate/`，无需另行下载。单题入口会验证文件 SHA256 和特征名称。请使用 Python 3.10 与 `requirements-calibration.txt` 中固定的版本；只从可信仓库加载 `joblib` 文件。

对接方若已提供 `original_answer`，不需要在本服务中再次运行原始回答模型。Qwen3-VL 专家服务可使用官方模型 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；SAM3、PP-OCRv6 等专家可通过可选的 `expert_evidence` 接入。这些大模型的权重和服务须单独部署，不包含在本仓库。

只配置 Qwen3-VL 时，接口可以处理新图像，但不会自动获得 SAM3、OCR 和空间几何专家的观测。需要完整专家组合时，请按接口说明接入这些服务，并在目标系统联调。

冻结门控基于 Qwen2.5-VL-7B 的原始回答训练；换用另一种基座模型时，候选收益和误伤率可能改变，应在该基座上重新做成对评估。

无 GPU 服务时可运行 `python scripts/smoke_vpc3.py`，以模拟专家观测检查门控文件和接口格式。此命令不调用真实视觉模型。
