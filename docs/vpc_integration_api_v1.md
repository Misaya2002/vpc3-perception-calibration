# VPC3 原生答案与证据输出接口

本文定义 VPC3 的稳定输入输出字段。`scripts/run_vpc3.py` 是单题 JSON 命令行入口；`scripts/serve_vpc3.py` 提供本机 HTTP `POST /calibrate`；`woodpecker_qwen.vpc3_integration.VPC3Calibrator` 可供 Python 系统直接调用。批处理脚本 `scripts/apply_native_sklearn_gate.py` 仍接受预先冻结的专家候选。入口不会自动启动 SAM3、OCR 等模型。

## 调用方输入

面向上层集成的单题输入约定：

```json
{
  "request_id": "example-001",
  "image": {"source": "path", "value": "/data/images/table.jpg"},
  "question": "How many red cups are on the table?",
  "original_answer": "2"
}
```

`original_answer` 必须是基座模型对同一图像、同一问题生成的答案。`image.source` 支持 `path`（本机可读文件）或 `base64`（图片字节）；入口不下载远程 URL。问题路由和专家证据生成后，门控才选择保留原答案或采纳候选答案。

HTTP 调用时，`path` 指 VPC3 服务所在机器的路径；如调用方与 VPC3 不在同一机器，使用 `base64` 传图。服务默认监听 `127.0.0.1:8765`，`GET /health` 返回健康状态。

```bash
python scripts/run_vpc3.py --input request.json --output response.json \
  --qwen3-base-url http://localhost:8000/v1
```

```bash
python scripts/serve_vpc3.py --qwen3-base-url http://localhost:8000/v1
curl -X POST http://127.0.0.1:8765/calibrate \
  -H 'Content-Type: application/json' --data-binary @request.json
```

对接方若已有专家服务，可在输入中附加可选的 `expert_evidence` 对象。本次冻结门控实际训练候选支持 `presence`、`color`、`siglip2_color`、`spatial`、`count_anything`、`ocr`、`qwen3_vl`；值采用 `woodpecker_qwen/native_experts.py` 所读取的相应证据结构。`countgd`、32B 和 BLIP 候选未进入本轮门控训练，单题入口会拒绝这些来源。Qwen3-VL 服务可与其他专家证据同时使用。未配置 Qwen3-VL 且未提供任何专家证据时，入口会报错。

例如，SAM3 存在性服务已检测到目标时，可以传入：

```json
"expert_evidence": {
  "presence": {
    "objects": [{"label": "dog", "score": 0.91, "bbox": [10, 20, 80, 110]}]
  }
}
```

入口只消费这些观测，不会下载或启动 SAM3。完整专家组合应由对接方把各专家服务的输出映射到上述字段；单独运行 Qwen3-VL 的增益须重新测量。

对接其他专家时，最小证据结构如下（均放在 `expert_evidence` 对应键下）：

| 键 | 必需字段 | 候选产生条件摘要 |
| --- | --- | --- |
| `presence` | `objects: [{label, score, bbox}]` | 问题路由允许、原答案为 No，目标匹配且分数至少 0.45 |
| `color` | `colors: [{target, color, confidence, bbox}]` | 颜色候选有效、分数至少 0.45 |
| `siglip2_color` | `question`, `candidates: [{answer, score, margin, bbox, crop_box}]` | 问题完全一致、分数至少 0.25 且 margin 至少 0.02 |
| `count_anything` | `question`, `status`, `query_matches_question_target`, `query`, `count`, `points`, `score` | query 必须等于问题路由解析的目标；`points` 长度须等于 count |
| `spatial` | `candidates: [{question_id, status, syntax_verified, relation, anchor_box, candidate_box, answer, score}]` | status 为 `geometry_verified`，关系与两个检测框几何位置一致 |
| `ocr` | `ocr: [{text, confidence, bbox}]` | 单一高置信文本行，或与 Qwen3 候选完全一致的 OCR 行 |
| `qwen3_vl` | `question`, `answer`, `status`, `model_revision` | 问题完全一致，status 为 `complete`，答案符合该题原生类型 |

门控没有收到有效候选、候选未过阈值，或提供的来源不在该门控训练支持内时，会保留原答案或拒绝输入。专家分数只用于构造候选；是否采纳由冻结门控决定。

批处理门控入口使用冻结的 `proposals.jsonl` 和门控文件：

```bash
python scripts/apply_native_sklearn_gate.py \
  --proposals /path/to/frozen_proposals \
  --gate /path/to/frozen_gate \
  --output-dir /path/to/vpc3_output
```

候选文件含 `question_id`、`image`、`question`、`raw_answer`、`answer_type`、`perceptual_task` 和 `proposals`。运行不读取参考答案或测试标签。候选证据由前序路由、专家与工具流水线生成。

## 输出

每题输出一行 JSONL，只保留调用和核验最关键的信息：

```json
{
  "request_id": "example-001",
  "original_answer": "2",
  "calibrated_answer": "3",
  "changed": true,
  "evidence": {
    "source": "count_anything",
    "observation": "red cups: 3 个检测点",
    "candidate_answer": "3",
    "reason": "candidate_passed_gate"
  }
}
```

`evidence.observation` 是专家观测的简短摘要，例如检测到的目标/边框、空间关系、计数点数量或 OCR 文本。若门控保留原答案，`changed` 为 `false`；若有被拒绝的候选，证据会记录分数最高的候选及拒绝原因。未采纳候选不能解释成对原答案正确性的证明。

分数、阈值、答案类型和逐题内部路由不放在默认出参中；文件级 `PREDICTIONS_FROZEN.json` 仍记录方法版本、输入/模型/预测哈希和运行计数，供复现与审计。公开证据不包含绝对文件路径。

批处理输出目录还包含 `PREDICTIONS_FROZEN.json`，记录问题、候选、门控和预测文件的 SHA256 以及预测条数。`scripts/audit_native_five.py` 只在预测冻结并通过哈希、条数和题目对齐检查后读取评测标签，主指标是校准前后准确率差（百分点）。

## 当前能力边界

VPC3 保持五类原生答案空间：存在性 Yes/No、颜色词、空间题中的实体答案、整数计数和 TextVQA 自由文本。每个专家只能提出候选，门控负责决定是否采纳。仓库已包含冻结门控；完整专家组合还需要相应模型与证据生成服务。仅启用 Qwen3-VL 的实际性能不能按完整组合的开发集结果宣称。

接口示例用于说明字段，不代表示例题已经被模型处理或答案已被证实正确。性能应以冻结的成对评测为准。
