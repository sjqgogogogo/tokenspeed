# Prompt / output logprob 诊断

本文描述 Python `Engine.generate()` 与原生 HTTP `/generate` 使用的 prompt/output logprob 诊断契约。HTTP 路径需要同时升级 SMG gateway、gRPC proto 与 servicer，以支持 TokenSpeed complete 消息中的 input logprobs 和原生响应中的 top-K。CPU 测试通过不等于 GPU 数值、CUDA Graph 或性能已验收。

## 四类返回值

全部位于响应 `meta_info`，使用现有 SGLang tuple-list 格式：

| 字段 | 含义 | 每个位置的结构 |
| --- | --- | --- |
| `input_token_logprobs` | 原始 prompt 中真实 token 的条件 logprob；不是采样新 token | `[logprob, prompt_token_id, null]` |
| `input_top_logprobs` | 相同 prompt 位置的 top-K 候选分布 | `[[logprob, token_id, null], ...]` |
| `output_token_logprobs` | 实际生成 token 的 logprob，沿用原采样后端路径 | `[logprob, generated_token_id, null]` |
| `output_top_logprobs` | 每个生成位置的原始模型 top-K 候选分布 | `[[logprob, token_id, null], ...]` |

Top-K 与 input 实际 token 分数来自模型 logits 的 FP32 log-softmax，而不是只在 top-K 内重新归一化。不是完整词表 logits 导出。输出实际 token 分数保留原采样后端契约；对齐采集时不要启用 grammar、penalty 或 custom processor，以免将处理后的采样分数与原始模型分布混淆。Python API 沿用原有采样参数校验，并不额外禁止所有这些参数。采样温度、top-p/top-k 决定所选 token，不改变诊断原始 Top-K 分布的定义。

`logprob_start_len=s` 是 **prompt token 索引**：

- `s=0`：返回 N 个 input 位置。第 0 个真实 token 为 `[null, prompt_ids[0], null]`；其 Top-K 为 `null`，因为没有左侧上下文。
- `0<s<N`：返回从第 s 个真实 prompt token 开始的 N-s 项；第一个分数来自 source row s-1。
- `s=-1`：只采集 output，input 两字段为空列表。
- chunk source row j 预测下一位置 `chunk_start+j+1`。跨 chunk 只右移一次；最后 prompt source row 预测 output 第一个 token，不再计入 input。
- Python API 支持 `K=0,s>=0`，只采集 input/output 实际 token 分数；未请求的 Top-K 字段可能省略，调用方可用 `meta_info.get(key, [])`，不能用占位分数伪造数据。

执行结果在 GPU/CPU 提交边界转为 CPU tensors 后才进入状态拼装。新增 prompt/Top-K 诊断仅支持非流式请求：内部提交帧仍按位置累计，最终通过现有非流式 collector 的 latest-wins 语义返回完整结果。`InputProcessor` 拒绝新增诊断的 `stream=True`。本扩展不改 collector，也不修复或改变普通流式输出的既有累计契约。

## 开关与范围

启动时同时添加：

```bash
--enable-input-logprobs --enable-output-logprobs
```

新 `--enable-input-logprobs` 默认关闭；仅启用 output 不会隐式开启 input。现有 `--enable-output-logprobs` 不是“导出全部 logits”开关，原本只驱动实际生成 token 的 logprob 计算。本扩展仍要求该开关，以复用既有采样分数路径。

eager 诊断请求的验收配置：

```bash
--enforce-eager --disable-overlap-schedule \
--disable-prefix-caching --disable-kvstore
```

此外要求 `stream=False`、单体服务、无 speculative decoding、PP=1、Attention DP/CP=1、Dense DP 关闭、`dp_sampling=False`、非多模态输入。TP 可保留。无诊断请求的普通生成及原有 sampled-output-only 请求（`start=-1,K=0`，包括 stream）不受新增 gate 限制。本页的 CPU/eager 回归不证明 CUDA Graph / overlap 支持；相关 gate、图重放证据和逐项数值验收需要独立核验。

## 直接使用 Python 引擎

使用现有入口，不经过 HTTP 或 SMG：

```text
Engine.generate(..., stream=False)
  → LLM.generate
  → AsyncLLM.generate_request
  → 现有非流式 collector
```

在既有模型、TP 和后端配置上设置 Python 参数 `enable_input_logprobs=True`、`enable_output_logprobs=True`、`enforce_eager=True`、`disable_overlap_schedule=True`、`enable_prefix_caching=False`、`disable_kvstore=True`，其他限制见上节。随后调用已创建的 `engine`：

```python
# input_ids 是已核实的原始整数 token IDs，不对文本重复编码。
result = engine.generate(
    input_ids=input_ids,
    sampling_params={
        "max_new_tokens": 128,
        "temperature": 0,
        "top_p": 1.0,
        "ignore_eos": False,
        "no_stop_trim": True,
        "skip_special_tokens": False,
    },
    return_logprob=True,
    logprob_start_len=0,
    top_logprobs_num=10,
    return_text_in_logprobs=False,
    logprob_format="sglang",
    stream=False,
)
meta = result["meta_info"]
output_ids = result["output_ids"]
assert len(meta["input_token_logprobs"]) == len(input_ids)
assert len(meta["input_top_logprobs"]) == len(input_ids)
assert len(meta["output_token_logprobs"]) == len(output_ids)
assert len(meta["output_top_logprobs"]) == len(output_ids)
assert meta["input_token_logprobs"][0] == [None, input_ids[0], None]
assert meta["input_top_logprobs"][0] is None
assert [row[1] for row in meta["input_token_logprobs"]] == input_ids
assert [row[1] for row in meta["output_token_logprobs"]] == output_ids
```

示例要求单样本非空 `input_ids`、`start=0`、`K>0`。调用方应另外检查 `finish_reason` 没有 abort、分数有限，以及每个可比较位置有完整 K 个候选；不要把请求失败、缺失字段或非首位置的 None 算作正常结果。`no_stop_trim=True` 和 `skip_special_tokens=False` 需要调用方显式传入，直接引擎调用不会自动补这两个参数。

原生 full-prefill 对齐时，把整数 `prompt_ids + output_ids` 作为新请求的 `input_ids`，`max_new_tokens=1` 只用于完成请求，该额外生成 token 不纳入 full-prefill 比较。本扩展不包含 router replay，也不自动修改采集脚本或比较指标。

`Engine.async_generate()` 也透传相同诊断参数。

## 原生 HTTP `/generate`

HTTP 请求继续走 SMG，不增加 Chat/Completion 扩展。部署时需要成套更新 gateway、proto 和 servicer；仅更新 TokenSpeed 不会补齐传输链路。SMG 必须支持 `GenerateComplete.input_logprobs`，保留显式的 `logprob_start_len=0`，并返回 `meta_info.input_top_logprobs` 和 `meta_info.output_top_logprobs`。现有实际输出 token 分数继续使用 `--enable-output-logprobs` 的原路径。

在上述启动配置下，请求示例：

```json
{
  "input_ids": [1, 2, 3],
  "sampling_params": {
    "max_new_tokens": 8,
    "temperature": 0,
    "top_p": 1.0,
    "no_stop_trim": true,
    "skip_special_tokens": false
  },
  "return_logprob": true,
  "logprob_start_len": 0,
  "top_logprobs_num": 5,
  "stream": false
}
```

示例 token IDs 必须替换为模型 tokenizer 对应的真实输入。CI 应使用返回的整数 `output_ids` 拼接原始 `input_ids`，再执行 full-prefill；不能把生成文本重新分词作为对齐依据。SMG 原有 chosen-token 返回项为 `[logprob, token_id]`，新增 top-K 项带空 text 槽 `[logprob, token_id, null]`；比较程序读取前两个槽即可。首 prompt token 的 chosen 分数和候选分布均为空，实际零分数则保留为数值零。`K=0` 时不制造 top-K 分数。

本次诊断支持并发请求，但不解除 prefix-cache、overlap 等诊断限制。串行、每次 flush 的 bitwise 回归，以及较高并发的容差比较，仍由外层测试编排控制。真实 GPU 上的 bitwise 稳定性和 prefill/decode 数值容差必须单独验收。

## CPU 回归与 GPU 验收边界

```bash
python -B -m pytest -q -p no:cacheprovider \
  test/runtime/test_prompt_top_logprobs_pipeline.py \
  test/runtime/test_prompt_logprob_wire.py \
  test/runtime/test_engine_logprobs.py \
  test/runtime/execution/test_top_logprob_capture.py \
  test/runtime/execution/test_graph_logprob_capture.py
```

测试只读取当前仓库内的生产方法/函数，普通 logits 以独立矩阵数学期望校验，不依赖仓库外的旧源码副本。直接入口测试执行生产 Engine 方法及同步适配器，仅替代请求依赖和实际生成端；不等价于启动真实 GPU 引擎。通过数量应以当前执行输出为准。

覆盖目标：prompt 首 None、每个 start 索引、跨 chunk/重算去重、真正非 None input token 概率、不同 K 混批、提前 EOS、最终 prompt row/output 首 token 边界、非流式累计、诊断 stream 拒绝与普通 sampled-output stream 放行、输出晚到不重复、K0、缺 payload 报错、CPU tensor 边界，以及直接入口的参数透传、四字段返回和错误传播。

GPU 下一步必须核实：

1. 短输入全量 logits 参考，逐项核对四类值与 IDs；关闭诊断时普通生成仍可用。
2. 输入超过 chunk 大小，确认完整 N 项和 output 第 0 项不重不漏。
3. 固定输入/seed，比较 diagnostic off/on 的实际输出及浮点差异；input 模式把 LM head 从末行选择扩为全 chunk，矩阵形状变化可能影响低精度舍入，不能只凭 CPU 测试宣称生成 bitwise 不变。
4. 检查 GPU 显存：完整 chunk × vocab logits / FP32 log-softmax 的峰值高于 sampled-output-only；从短输入小 chunk 开始，不直接投入长上下文高并发。
5. eager 成功后才在独立副本测试 CUDA Graph；不能把本轮限制静默取消。

## 显式 CUDA Graph 诊断

保留两项 logprob 开关和上述隔离条件，以 `--enable-logprob-graph --disable-prefill-graph` 替代 `--enforce-eager`。Prefill/chunk 仍走 eager；decode 使用原统一 refresh 和 sampler 路径。普通请求保留无快照的原 graph，诊断 Top-K graph 将原始 logits 拷入图池外的持久 FP32 buffer，再在 replay 后按 live batch 与每请求 K 收集。超出 capture ladder 或缺少诊断 graph 时明确失败。

每次诊断 decode replay 累计计数；验收启动必须显式设置 `TOKENSPEED_GRAPH_DEBUG=1`，才会发出 `LOGPROB_GRAPH_REPLAY`（rank/count/live_bs/padded_bs/variant/snapshot）。日志证明已发起 replay，不代替完成 fence 和成功响应；默认不逐步打印日志。验收需包含异构 K、普通/诊断交替、live BS1/2/3 pad4/4、跨 chunk、EOS 和 slot 复用。CPU mocked-graph 测试不证明 CUDA 数值或生命周期正确，GPU eager/graph 同输入对照必须独立执行。这是诊断功能，不用于公平吞吐基准。
