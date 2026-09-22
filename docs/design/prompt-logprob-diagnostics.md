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

保留服务原本的 CUDA Graph、overlap schedule、prefill graph 和 prefix cache 配置。非 MTP、非 PD 单体服务在开启 output logprobs 时自动准备可支持的 decode graph 快照；`--enable-logprob-graph` 保留为兼容开关，不再是必需参数。

请求 input logprobs（`return_logprob=True, logprob_start_len>=0`）时，Python 请求入口设置 C++ `RequestSpec.reuse_prefix_cache=False`。scheduler 在每次 admission（包括 retraction 后重入）绕过 L1/L2 前缀读取，为重新计算分配私有可写块；完成块仍按原规则发布缓存，其他请求可继续命中。只请求 output 分数/top-K 的请求继续正常复用缓存。需重新编译安装 `tokenspeed-scheduler`，仅更新 Python 源码不足以加入这个字段。

本期要求 prompt/top-K 请求 `stream=False`、单体服务、无 speculative decoding、PP=1、Attention CP=1、`dp_sampling=False`，请求内容为纯文本/token IDs。支持 attention TP、DP 及其组合，也允许同一 attention replica 内的 dense TP/DP 布局。模型具有视觉能力本身不是拒绝条件；实际图片、视频、音频和 embedding 输入仍拒绝。所有 return_logprob 请求（包括只取输出 token 分数）在 MTP/DSPARK/EAGLE3/DFLASH 或 PD 配置下明确报错。非投机、非 PD 的 sampled-output-only 请求仍可流式输出。GPU 数值与实际并行行为仍需验收，CPU 测试不能替代它们。

## 直接使用 Python 引擎

使用现有入口，不经过 HTTP 或 SMG：

```text
Engine.generate(..., stream=False)
  → LLM.generate
  → AsyncLLM.generate_request
  → 现有非流式 collector
```

在既有模型、TP 和后端配置上设置 Python 参数 `enable_input_logprobs=True`、`enable_output_logprobs=True`，保留 CUDA Graph、overlap 和缓存设置，其他范围限制见上节。随后调用已创建的 `engine`：

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

本次诊断支持并发、overlap、graph 和全局 prefix cache；input 分数请求的缓存绕过在 scheduler 中按请求执行。串行、每次 flush 的 bitwise 回归，以及较高并发的容差比较，仍由外层测试编排控制。真实 GPU 上的 bitwise 稳定性和 prefill/decode 数值容差必须单独验收。

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
5. 全局 cache 开启时，普通重复请求命中、input 分数请求不命中；混合 batch、L2、可控 retraction 下验证两者仍独立。
6. 在 graph 和 overlap 开启的部署上验证实际 replay 与在途请求行为；batch 超出捕获阶梯时保留引擎正常 eager 路由，报告真实执行路径。

## CUDA Graph 与 overlap 的数据所有权

Prefill 沿原 breakable graph 重放模型主体，随后在现有 eager logits tail 采集 prompt 分数；不会为 input logprobs 禁用 prefill graph。Decode 的每个 sampler/batch-size graph 保留普通版本，并用同一个 `_forward_step` 捕获带原始 logits 快照的版本。快照在采样修改 logits 前写入图池外的持久 FP32 buffer；graph 后按 live batch/每请求 K 取分数，padding 行不返回。

执行线程在同一 execution stream 上按 FIFO 提交：本轮 graph → 本轮 top-K 结果 → D2H → copy_event → 后续 forward。配置由 `PlannedForward` 捕获，分数在每轮创建的 `ModelExecutionResult` 中独立持有。控制线程只在原有 commit 边界通过 `PendingExecution.result()` 等待；不新增 GPU 同步、不为了诊断排空 overlap 队列。不要把静态 graph 缓冲直接交给异步 CPU 消费者。

原本符合 graph 条件的诊断请求必须实际 replay；缺少应有的诊断 graph 是错误。引擎原本不使用 graph 的形状（例如超过捕获阶梯）继续走相同 forward 的 eager 路由。`TOKENSPEED_GRAPH_DEBUG=1` 时 `LOGPROB_GRAPH_REPLAY` 记录 rank/count/live_bs/padded_bs/variant/snapshot/overlap_depth；depth 是配置，不单独证明实际在途深度，需结合调度 trace 或 profiler 验证 overlap。

新增的 `test/runtime/execution/test_logprob_overlap_cuda.py` 用真实 CUDA graph 连续重放和延迟 CPU 消费验证缓冲区生命周期（无模型下载）；无 CUDA 的机器会跳过。它不替代真实模型、attention、scheduler 和 HTTP 的端到端验收。诊断有额外计算/传输开销，不用于无诊断吞吐基准。

MTP 仅作可行性研究，当前拒绝所有投机解码下的 logprobs 请求，见 [后续 speculative logprobs](speculative-logprobs.md)。


## DSV4.1 完整输入行与 attention DP

请求输入评分的 extend/mixed forward 通过元数据字段 `full_prompt_logits`
保留完整 decoder 行，沿原 encoder → narrowing → decoder → LM head 执行。
该批次的 decoder view 为 identity，保留原 chunk 的 SWA prefix floor；
最后一行仍负责第一个 output token，不能再次作为 input 分数返回。
不请求输入评分时继续使用既有 CED 尾部裁剪。

这定义了完整因果 teacher-forcing 的输入评分路径，不能将旧的尾部裁剪
结果补零当作完整分数。官方 inference/model.py 的普通 Transformer.forward
也计算所有输入行。完整计算改变 decoder 上下文和 GEMM 行形状，不能声称
它与只计算尾部的普通生成 bitwise 一致；必须在 H200 上用参考模型核验
每个位置和跨 chunk 的数值。请求该能力也会增加计算量、logits 峰值显存。

控制平面每轮在原 DP all-gather 中额外交换 decoder token 数，并随不可变
`DpForwardMetadata` 传入 forward。常规请求取完成 prompt 的最后窗口/未完成
chunk 的一行，输入评分取所有行，decode 行保留。encoder 和 decoder 分别
使用各自计数，不在 forward 线程执行 D2H 或控制面 collective。
Attention TP 的完整行在跨 DP 的 MoE 之前按 TP rank 切分，MoE 后恢复；
空闲 DP rank 仍执行每层 MoE 通信。该扩展针对 V4.1 普通 dense EP 路径，
不改变 V4 的布局限制。

Decode 诊断 graph 允许 DP，使用已有跨 rank 共同 padded batch 决策；快照
只交给本地 live 请求。DSV4.1 在 DP 下的 split prefill graph 仍因缺少共同
decoder bucket 协议而关闭，prefill 走 eager，decode graph 与 overlap 保留。
DP1 的 split prefill graph 继续按现有容量选择：完整评分行超过 decoder
捕获容量时使用既有 eager decoder 回退，encoder graph 可继续重放。

## Dynamo 接入契约

非 PD 表示一个 worker 同时执行 prefill 和 decode，不表示所有分数来自
同一阶段。prompt 位置 i 的分数来自位置 i-1 的 prefill logits；首位置为
null。最后一次 prefill 产生 output 位置 0 及其分数，后续 decode 产生
剩余输出分数。长 prompt 跨多个 chunk，需要按绝对 token 位置合并。

Dynamo 适配器已按以下契约接入；其构建、请求示例与验收见 Dynamo 仓库的 `docs/backends/tokenspeed/logprobs.md`。Rust 前端与 Python 组件需要成套更新：

1. `build_logprob_kwargs` 保留显式 0，并映射输入/输出各自的 top-K 数量。引擎目前用一个 K 控制
   input/output；若 Dynamo 两个 K 不同，可以请求二者最大值，在响应侧
   各自裁剪，但不能更改概率的全词表归一化。
2. `generate` 对 prompt/top-K 请求使用引擎 `stream=False`，收齐后返回一个
   complete chunk（HTTP SSE 也会等待完整结果）；不能误以为 HTTP stream=false 会自动改变引擎调用。
   若要真正增量诊断，需要进一步扩展 TokenSpeed 的流式结果契约。
3. 适配器将输出实际 token 分数映射到 Dynamo 的
   `log_probs`，候选分数/IDs 映射到 `top_logprobs`。缺少请求的分数或 token ID 不匹配会明确报错。
   `_completion_delta_output` 裁剪 token IDs 时必须以同一索引裁剪分数，
   包括首 prefill 输出、终止帧、echoed prompt 和取消请求。
4. 输入分数不应混入输出 token 数组。Dynamo 的 `LLMEngineOutput` 已有
   `engine_data` 扩展，在 `nvext.engine_data.input_logprobs` 中承载 input IDs、
   chosen 分数、top-K 和起始位置。Chat/Completions 使用 `nvext.prompt_logprobs`
   请求输入评分，并自动开启该返回字段。
5. 加上请求转换、Rust 序列化/前端聚合和端到端测试，验证 null 与 0、
   不同 input/output K、跨 chunk、终止截断、批间不串数据。不依赖 SMG
   的 proto 修改：Dynamo 走直接 Python 引擎及自身协议。


### H200 验收配置与顺序

沿用已验证的 cu129 环境安装方式，重新安装以更新 scheduler 的请求字段。
无投机单体服务增加 `--enable-input-logprobs --enable-output-logprobs`，移除
`--speculative-algorithm DSPARK`。保留 overlap 和 decode CUDA graph，不加
`--enforce-eager` 或 `--disable-overlap-schedule`。先通过 Python Engine 直接
入口验收，先区分引擎执行问题与 Dynamo/SMG 协议问题，再做端到端验证。

8 卡至少覆盖 attention TP8/DP1、TP4/DP2、TP1/DP8，均使用 MoE EP8。
DP 场景应指定 dist-init-addr，并确认启动打印的实际 mapping。使用短输入
先核验四类返回值，再覆盖 127/128/129、chunk 边界两侧、不同请求长度与 K、
某个 DP replica 空闲和一个 replica prefill/另一个 decode。调试 trace 必须
显示实际 decode graph replay 和多个在途轮次；CPU mock graph 测试不证明
硬件上已经重放。

`test/runtime/execution/test_dp_logprob_contract.py` 覆盖 CPU 行数与切分协议；
`test/runtime/execution/test_logprob_overlap_cuda.py` 覆盖真实 CUDA graph 的
快照生命周期。模型权重、attention/MoE 数值、NCCL 多卡组合仍需目标机验收。
