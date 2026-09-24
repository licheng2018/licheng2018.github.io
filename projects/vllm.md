# vLLM Scheduler Analysis and Chunked Prefill Optimization

This project explores vLLM inference internals through source-code analysis, scheduler instrumentation, baseline benchmarking, and chunked prefill policy experiments. The work was carried out on a Google Colab T4, starting with small-model offline inference and server execution and progressing to scheduler modifications and correctness testing.

Source repo: [github](https://github.com/licheng2018/vllm)

## Project Goal

Connect serving performance with the internal decisions behind request scheduling and KV-cache management. The project focuses on tracing request execution, observing scheduler state at each step, and experimenting with fixed and adaptive chunked prefill policies.

![vLLM project overview: request tracing, scheduler and KV-cache analysis, instrumentation, benchmarking, and chunked prefill policy experiments](../assets/projects/vllm/project-overview.png)

![vLLM offline request lifecycle from LLM.generate() through input processing, EngineCore, WAITING and RUNNING states, scheduler output, ModelRunner, and output processing](../assets/projects/vllm/request-lifecycle.png)

![vLLM 0.29.0 scheduler internals: token and input budgets, running requests, waiting admission, KV allocation and preemption, SchedulerOutput, and execution feedback](../assets/projects/vllm/scheduler-internals.png)

![vLLM KV cache memory management: contiguous allocation fragmentation, paged block mapping, token and memory budgets, request growth, and preemption](../assets/projects/vllm/kv-cache-memory-management.png)

## Experimental Setup

| Component | Setting |
|---|---|
| Environment | Google Colab |
| GPU | NVIDIA T4 |
| Inference framework | vLLM |
| Execution modes | Small-model offline inference and inference server |
| Performance measurements | TTFT, TPOT, throughput, and KV-cache pressure |
| Policy experiments | Fixed and adaptive chunked prefill |

## Implementation Milestones

| Stage | Work | Outputs |
|---|---|---|
| Environment and baseline | Set up vLLM and run small-model offline inference and server execution. | Runnable environment and baseline commands. |
| Request lifecycle | Trace the request path through API, EngineCore, Scheduler, and ModelRunner. | Request lifecycle diagram and source-code notes. |
| Scheduler analysis | Study waiting/running states, token budgets, and continuous batching. | Scheduler flow diagram and key-function explanations. |
| KV cache and PagedAttention | Analyze block allocation, cache pressure, and preemption. | KV-cache flow diagram and key-function explanations. |
| Instrumentation | Add scheduler instrumentation to record state at each scheduling step. | `scheduler_trace.jsonl` and a trace script. |
| Baseline benchmarking | Measure TTFT, TPOT, throughput, and KV-cache pressure. | Baseline results table and two to three plots. |
| Policy modification | Modify chunked prefill policy, starting with fixed chunks and extending to adaptive chunks. | Scheduler patch and correctness test. |

## Request Lifecycle and Scheduler Analysis

This walkthrough follows the code and saved outputs in [vLLM_Foundation.ipynb](https://github.com/licheng2018/vllm/blob/main/vLLM%20Setup%2C%20Request%20Inspection%2C%20and%20Baseline%20Benchmarking/vLLM_Foundation.ipynb). The recorded environment is **vLLM 0.29.0**, PyTorch **2.13.0+cu130**, Python **3.13.15**, and a **Tesla T4 with 14.56 GiB VRAM**. The offline example uses **Qwen/Qwen2.5-1.5B-Instruct**, FP16, `max_model_len=4096`, and `gpu_memory_utilization=0.70`.

The investigation combines an actual inference run with inspection of the installed Python source. **Source verified** below means that the notebook printed the relevant implementation. **Runtime observed** means that a saved execution output shows the behavior. These are different levels of evidence: printing a branch establishes what the code does when reached, but does not establish that a particular request took that branch.

Notebook cell references below use **one-based physical cell positions**, counting both Markdown and code cells, rather than execution counts. Code snippets are shortened inspection commands or selected implementation excerpts; omitted branches are not intended as a replacement implementation.

### Step 1 — Establish a working request before tracing internals

**What I wanted to see.** Can a small model accept prompts and return outputs on the T4, using the same public API that the source investigation will follow?

**How the code does it.** Notebook cells 14–15 write and execute a separate Python script:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.70,
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
outputs = llm.generate(prompts, sampling_params)
```

**What the saved output establishes.** Runtime observed: engine initialization identifies V1 and v0.29.0, and the initial batch reaches `Processed prompts: 100% 2/2`. Initialization also reports chunked prefill with `max_num_batched_tokens=8192`. A Triton JIT warning appears during inference, so the first run includes compilation activity. These logs establish that inference runs; they do not expose individual WAITING/RUNNING transitions or prove a continuous-batching speedup.

### Step 2 — Locate the public entry point: LLM.generate()

**What I wanted to see.** Where is the public API implemented, and does it execute the model itself or delegate to another layer?

**How I inspected it.** Notebook cells 40–41:

```python
import inspect
from vllm import LLM

print(inspect.getfile(LLM.generate))
print(inspect.getsource(LLM.generate))
```

**What I found.** Source verified: the installed method lives in `vllm/entrypoints/llm.py`. It checks the model's runner type, supplies default sampling parameters if needed, and calls `self._run_completion(...)`. The forwarded arguments include prompts, sampling parameters, `RequestOutput` as the output type, progress-display settings, LoRA, priority, and preprocessing options. This identifies the public wrapper and its next call; there is no model forward pass in the printed method.

### Step 3 — Open the offline utilities layer

**What I wanted to see.** What happens between the public API and per-request engine submission?

**How I inspected it.** Notebook cells 42–43:

```python
print(inspect.getfile(LLM._run_completion))
print(inspect.getsource(LLM._run_completion))
print(inspect.getsource(LLM._add_completion_requests))
```

**What I found.** Source verified: these helpers live in `vllm/entrypoints/offline_utils.py`. `_run_completion()` first calls `_add_completion_requests()` and then `_run_engine()`. Submission and engine driving are separate operations. `_add_completion_requests()` normalizes prompts, parameters, LoRA requests, and priorities into matching sequences, then supplies `_preprocess_cmpl_one(...)` results to `_render_and_add_requests()`.

The saved source therefore expands “offline utilities” into concrete functions. It shows the preprocessing call site, but does not separately print the implementation of `_preprocess_cmpl_one()` or `_run_engine()`; their full internals are not demonstrated by these cells.

### Step 4 — Submit each prompt and create its initial request ID

**What I wanted to see.** Where does a batch become individual requests, and where is the first request ID assigned?

**How I inspected it.** Notebook cells 44–45 print `LLM._render_and_add_requests` and `LLM._add_request`. Selected lines from the latter are:

```python
params.output_kind = RequestOutputKind.FINAL_ONLY
request_id = str(next(self.request_counter))
return self.llm_engine.add_request(
    request_id, prompt, params,
    lora_request=lora_request, priority=priority,
)
```

**What I found.** Source verified: `_render_and_add_requests()` loops over prompts, pairs each with its parameters, and collects the returned IDs. If submission fails after some requests were added, it aborts those already submitted. `_add_request()` creates a counter-based string ID and forwards one request to `LLMEngine.add_request()`.

`FINAL_ONLY` explains why this offline call collects completed results rather than exposing an HTTP token stream. This is the **initial** ID: the next layer also calls `assign_request_id()`, whose implementation was not inspected here, so the counter string should not be assumed to remain unchanged throughout the system.

### Step 5 — Inspect LLMEngine.add_request() and output registration

**What I wanted to see.** What does the engine do before handing a request to EngineCore, and how will returned tokens be associated with the caller?

**How I inspected it.** Notebook cells 46 and 48 print `LLMEngine.add_request()`. The import through `vllm.engine.llm_engine` resolves to an implementation in `vllm/v1/engine/llm_engine.py`. For the ordinary single-output path, the printed sequence is:

```python
request = self.input_processor.process_inputs(...)
self.input_processor.assign_request_id(request)
# For n == 1:
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

**What I found.** Source verified: input processing happens before core submission, and output-side request state is registered before the request is forwarded. The method also contains a separate parent/child fan-out path for `n > 1`. This makes the output processor part of request setup, not merely a formatter invoked at the very end.

### Step 6 — Find the EngineCoreRequest construction site

**What I wanted to see.** At what exact point does processed input become a structured engine request, and what information travels with it?

**How I inspected it.** Notebook cell 47:

```python
from vllm.v1.engine.input_processor import InputProcessor

print(inspect.getfile(InputProcessor.process_inputs))
print(inspect.getsource(InputProcessor.process_inputs))
```

**What I found.** Source verified: `vllm/v1/engine/input_processor.py` explicitly returns an `EngineCoreRequest(...)`. Before that, it validates parameters and model inputs, accepts an already rendered `EngineInput` or renders a raw prompt, extracts token IDs or embeddings, and clones and updates generation parameters.

| Request information | Why it is needed downstream |
|---|---|
| `request_id` | Identify the request across engine components. |
| `prompt_token_ids`, optional embeddings | Represent model input after preprocessing. |
| `sampling_params` or `pooling_params` | Define the requested generation or pooling behavior. |
| `arrival_time`, `priority` | Preserve timing and scheduling metadata. |
| LoRA, multimodal, routing, and tracing fields | Carry optional execution context. |

The object construction site and field assignments are visible in the saved source. The notebook does not print a live `EngineCoreRequest` instance with all these fields populated for a specific prompt.

### Step 7 — Distinguish EngineCoreRequest from the internal Request

**What I wanted to see.** Does the scheduler receive the same object constructed by the input processor?

**How I inspected it.** Notebook cell 49 prints `EngineCore.add_request()`, and cell 56 prints the complete `Request` class:

```python
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request

print(inspect.getsource(EngineCore.add_request))
print(inspect.getsource(Request))
```

**What I found.** Source verified: `EngineCore.add_request(self, request: Request, ...)` accepts the internal `Request` type and calls `self.scheduler.add_request(request)`. The printed `Request.from_engine_core_request()` factory constructs that internal object from an `EngineCoreRequest`.

**What remains only partially traced.** The notebook's “core client” inspection reprints `LLMEngine.add_request()`; it does not print the concrete client transport or the EngineCore preprocessing call site that invokes this factory. The conversion factory and both sides of the handoff are visible, but the complete client/IPC bridge is not established by these cells. The lifecycle diagram compresses that bridge.

### Step 8 — Separate WAITING initialization from waiting-queue insertion

**What I wanted to see.** When does a request become WAITING, and when does the scheduler actually store it?

**How the inspected code implements it.** Notebook cells 50, 55, and 56 expose three distinct pieces:

```python
# Request.__init__:
self.status = RequestStatus.WAITING

# Scheduler.add_request, new-request branch:
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request

# Scheduler.__init__:
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(self.policy)
self.running: list[Request] = []
```

**What I found.** Source verified: the default status is assigned during object construction, before scheduler registration. Registration then enqueues the request and indexes it by ID; it does not launch a GPU kernel. The waiting queues are created using the configured scheduling policy. Requests with structured-output requirements can receive a specialized waiting status, so plain WAITING describes the ordinary path rather than every possible request.

The notebook shows the state assignment and enqueue call, but contains no timestamped observation of a particular request entering the queue.

### Step 9 — Understand the token state that drives scheduling

**What I wanted to see.** What does “remaining work” mean to the scheduler, and why can the same mechanism cover both prefill and decode?

**How I inspected it.** Notebook cell 56 prints `Request`, including its fields and properties.

| Field or property | Meaning in the inspected implementation |
|---|---|
| `num_prompt_tokens` | Length of the prompt input. |
| `num_tokens` | Length of the accumulated non-speculative token sequence. |
| `num_tokens_with_spec` | Accumulated tokens plus speculative candidate tokens. |
| `output_token_ids` | Read-only view of generated output token IDs. |
| `num_computed_tokens` | Progress accounted for by the scheduler, including optimistically scheduled work. |
| `num_in_flight_tokens` | Scheduled tokens whose execution output has not yet been processed. |

**What I found.** Source verified: token counts are not interchangeable. In particular, the `Request` source explicitly describes in-flight work as being included optimistically in `num_computed_tokens`. Prefill and decode can be understood through the gap between available tokens and accounted-for progress, rather than by treating every scheduling step as exactly one generated token.

### Step 10 — Read the decisions made by Scheduler.schedule()

**What I wanted to see.** Which requests get work in this round, how much work do they receive, and what limits that decision?

**How I inspected it.** Notebook cell 51 prints the complete method; cells 57 and 64 filter its token-budget and chunk-policy logic. The running-request calculation includes:

```python
token_budget = self.max_num_scheduled_tokens
input_budget = self.scheduler_config.max_num_batched_tokens

num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
# After applying the long-prefill threshold:
num_new_tokens = min(
    num_new_tokens, token_budget, input_budget - draft_slots
)
```

**What I found.** Source verified: this version distinguishes a scheduling-token budget from the input-batch budget. `max_num_running_reqs` is set from `max_num_seqs`; `max_num_scheduled_tokens` uses its explicit configuration value or falls back to `max_num_batched_tokens`. The ordinary scheduling flow considers running requests before admitting waiting requests, while accounting for additional pause, dependency, and execution constraints.

The scheduler determines request selection, per-request token counts, and KV allocation. It records allocations in `num_scheduled_tokens[request_id]` and reduces the remaining budgets. The source explains how batching can change between iterations; the notebook does not measure a live mixed-request batch composition in these inspection cells.

### Step 11 — Identify the exact WAITING → RUNNING transition

**What I wanted to see.** What must succeed before a queued request becomes RUNNING, and does that happen before or after model execution?

**How I inspected it.** Notebook cell 58 filters waiting admission, allocation, and status changes from `Scheduler.schedule()`; cell 65 prints the chunked-prefill admission condition. For a waiting request, the method computes remaining work using `request.num_tokens - num_computed_tokens`, applies the request budget and chunk limits, and attempts `kv_cache_manager.allocate_slots(...)`.

Selected lines from the successful admission path are:

```python
self.running.append(request)
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

**What I found.** Source verified: RUNNING is assigned inside scheduling, after successful admission and allocation, before the returned plan is executed. It does not mean prefill has finished, the first output token exists, or the GPU has completed this round. The source also shows that disabling chunked prefill can prevent admission when the required work exceeds the available request budget.

The assignment site is confirmed. A per-request WAITING duration or actual transition timestamp is not present in the saved inspection output.

### Step 12 — Check the KV-allocation failure and preemption path

**What I wanted to see.** What happens when token budget exists but KV-cache capacity prevents execution?

**How I inspected it.** Notebook cell 59 searches `Scheduler.schedule()` for `allocate_slots`, `new_blocks`, and preemption-related lines. Cell 68 prints `KVCacheManager.allocate_slots()`, and cell 63 includes the tail of the scheduler's preemption handling.

**What I found.** Source verified: allocation can return `None`. In the running-request flow, the scheduler can select a victim, call `_preempt_request(...)`, and retry allocation; waiting admission also has an allocation-failure path. The inspected preemption tail puts a request back with `self.waiting.prepend_request(request)` and records its ID for worker notification.

This establishes the control path connecting memory availability to scheduling. It does **not** establish that the notebook's baseline run triggered preemption, how often it occurred, or how much latency it caused. Those claims require runtime traces or counters. Waiting-queue membership should also be distinguished from the request's status enum: a resumed preempted request is not simply a newly arrived WAITING request.

### Step 13 — Inspect the SchedulerOutput handed toward execution

**What I wanted to see.** Does the worker receive the original prompt request again, or a plan describing only this iteration?

**How I inspected it.** Notebook cells 83–84 follow block IDs into the output and print the data classes:

```python
import vllm.v1.core.sched.output as sched_output_module

for cls in (
    sched_output_module.NewRequestData,
    sched_output_module.CachedRequestData,
    sched_output_module.SchedulerOutput,
):
    print(inspect.getsource(cls))
```

**What I found.** Source verified: `SchedulerOutput` is a per-iteration execution plan with distinct representations for newly scheduled requests and requests already known to the worker.

| Payload | What the printed data classes contain |
|---|---|
| `scheduled_new_reqs` | `NewRequestData`: request ID, prompt tokens/embeddings, sampling or pooling parameters, block IDs, and computed-token count, with optional context. |
| `scheduled_cached_reqs` | `CachedRequestData`: IDs, resumed-request IDs, token updates, block updates, and progress information for previously scheduled requests. |
| `num_scheduled_tokens` | Mapping from each request ID to its scheduled token count. |
| `total_num_scheduled_tokens` | Total token work assigned in this round. |
| `finished_req_ids` | IDs whose worker-side cached state can be removed. |
| Additional metadata | Optional speculative, encoder, connector, block-zeroing, and block-copy information. |

The class definitions and construction site are visible. No concrete runtime `SchedulerOutput` instance is printed, so this is a verified schema and producer path, not an observed batch example.

### Step 14 — Locate the progress update before GPU completion

**What I wanted to see.** Exactly where does `num_computed_tokens` increase, and what would that counter mean in a trace?

**How I inspected it.** Notebook cells 62–63 search the scheduler class and print the relevant context:

```python
# Scheduler._update_after_schedule:
for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.num_in_flight_tokens += num_scheduled_token
```

**What I found.** Source verified: `schedule()` constructs its output, calls `_update_after_schedule(scheduler_output)`, and then returns it. The original progress values must first be captured for input preparation; the scheduler then advances its own accounting so future work can be scheduled. Later output handling can correct that accounting, including speculative-token rejection.

This changes how instrumentation should be interpreted: a counter sampled after `schedule()` can already include work that has not completed on the GPU. A meaningful trace must record both the observation point and in-flight work.

### Step 15 — Follow request data into ModelRunner and KV addressing

**What I wanted to see.** How does the ModelRunner consume the scheduler's request and block data, and how do logical token positions connect to physical KV storage?

**How I inspected it.** Notebook cells 85–86 search and print `GPUModelRunner` source around `scheduled_new_reqs`, `scheduled_cached_reqs`, `block_ids`, and `block_table`. Cells 87–91 follow `compute_slot_mapping()` into its implementation.

```python
import vllm.v1.worker.gpu_model_runner as runner_module
import vllm.v1.worker.block_table as block_table_module

print(inspect.getsource(runner_module.GPUModelRunner))
KernelClass = type(block_table_module._COMPUTE_SLOT_MAPPING_KERNEL)
print(inspect.getsource(KernelClass))
```

**What I found.** Source verified: new request data is used to build `CachedRequestState`. For known requests, the runner updates token/progress state and extends block IDs; resumed requests can replace their block mapping. The printed code also updates the input batch's block table, commits it, and computes slot mappings.

The investigation initially found only the `MultiGroupBlockTable` wrapper. Following its module located the concrete `BlockTable`, whose `compute_slot_mapping()` invokes `_COMPUTE_SLOT_MAPPING_KERNEL` with token positions, request offsets, GPU block-table data, and block-size parameters. Directly calling `inspect.getsource()` on the kernel object failed; inspecting `type(kernel)` successfully exposed `ComputeSlotMappingKernel` and its Triton implementation.

This is a concrete source-level connection from scheduler-selected KV block IDs to worker-side memory addressing. The notebook does not dump actual GPU slot mappings or profile a forward pass here, and inspecting this runner class alone does not establish which optional runner implementation every experiment used.

### Step 16 — Follow generated tokens back to request state and caller output

**What I wanted to see.** How are model results reconciled with scheduled work, and how does the caller receive a finished result?

**How I inspected it.** Notebook cells 60–61 locate and print `Scheduler.update_from_output()`. The printed implementation reads `model_runner_output.sampled_token_ids`, reconciles in-flight counts, calls `_update_request_with_output(...)` when token IDs are available, handles stopped requests, and constructs `EngineCoreOutput` objects carrying new tokens and finish information.

The original inference script in notebook cell 14 also accesses the public result fields:

```python
for output in outputs:
    candidate = output.outputs[0]
    # Written to the script's inspection file:
    output.request_id
    output.prompt_token_ids
    output.finished
    candidate.text
    candidate.token_ids
    candidate.finish_reason
```

**What I found.** Source verified: output handling updates the request lifecycle and emits core results, with a finished-request path that invokes resource cleanup. The earlier `FINAL_ONLY` setting explains the offline caller's completed-result contract. Runtime progress confirms that the initial batch completed.

**What is not shown.** The script writes detailed `RequestOutput` fields and timing results to `/content/request_output_inspection.txt`, but that file's contents are not displayed in this notebook's saved outputs. The output processor's full detokenization implementation and `_run_engine()` loop are also not printed. Therefore, the notebook demonstrates successful generation and several output-handling boundaries, but not a complete token-by-token execution trace or the exact returned text and finish reasons in the inspection file.

### What this investigation establishes

| Question | Evidence-based conclusion |
|---|---|
| Where does the engine request form? | `InputProcessor.process_inputs()` explicitly constructs `EngineCoreRequest`. |
| Where does the scheduler's internal object form? | The `Request.from_engine_core_request()` factory is printed; its core-client/preprocessing call site is not traced in these cells. |
| When does WAITING begin? | Default status initialization occurs in `Request.__init__`; scheduler insertion is a separate operation. |
| When does RUNNING begin? | The successful admission path sets it inside `Scheduler.schedule()`, before execution. |
| What does a round decide? | Request selection, token counts, admission/defer/preemption decisions, and KV allocation under capacity constraints. |
| What does ModelRunner consume? | Per-iteration `SchedulerOutput` data used to update cached request state, the batch, and KV block mappings. |
| Was every transition observed live? | No. Most evidence is saved source inspection; per-request event timing and actual scheduled-batch payloads remain for instrumentation. |


## Understanding vLLM Scheduler Internals

This section follows the scheduler-internals chapter of [vLLM_Foundation.ipynb](https://github.com/licheng2018/vllm/blob/main/vLLM%20Setup%2C%20Request%20Inspection%2C%20and%20Baseline%20Benchmarking/vLLM_Foundation.ipynb), in its original investigation order. The lifecycle walkthrough above follows a request across components; here the focus is how **one scheduling round distributes limited token and KV-cache resources across requests**.

The evidence comes from saved inspection outputs for **vLLM 0.29.0**, primarily notebook cells **55–65**. Cell numbers count all physical notebook cells from one. The snippets below reproduce the inspection approach or selected source excerpts. They are not standalone scheduler implementations. These cells inspect source rather than record live request-state transitions.

### 1. Inspect scheduler initialization: queues, policy, and capacity

**Question.** Where are requests stored, and which configuration values constrain an iteration?

**What I did.** The chapter starts by locating and printing `Scheduler.__init__()` in notebook cell 55:

```python
import inspect
from vllm.v1.core.sched.scheduler import Scheduler

print(inspect.getfile(Scheduler.__init__))
print(inspect.getsource(Scheduler.__init__))
```

**Implementation found.** The scheduler maintains an ID-to-request dictionary, policy-aware waiting queues, and a running list:

```python
self.requests: dict[str, Request] = {}
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(self.policy)
self.running: list[Request] = []
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
```

`self.policy` comes from `SchedulingPolicy(self.scheduler_config.policy)`. `max_num_scheduled_tokens` uses its explicitly configured value when present, otherwise falling back to `max_num_batched_tokens`. These values become two distinct budgets inside `schedule()`:

| Constraint | What it limits |
|---|---|
| `max_num_running_reqs` | Concurrent running-request capacity. Specialized streaming sessions also affect the admission count. |
| `token_budget` | Remaining scheduled-token work in this iteration. |
| `input_budget` | Remaining input-batch capacity, including reserved draft slots when applicable. |
| KV-cache availability | Whether the selected work can obtain the required memory blocks. |

**Did the inspection answer the question?** Yes, at source level. It located the containers and configuration bindings. It did not print a live scheduler's queue lengths or all resolved configuration values. The initialization log elsewhere in the notebook reports `max_num_batched_tokens=8192`; that is a configured limit, not evidence that every round schedules 8,192 tokens.

### 2. Inspect Request token state: distinguish input, output, and progress

**Question.** How does the scheduler calculate work remaining for a request?

**What I did.** Notebook cell 56 prints the `Request` class:

```python
from vllm.v1.request import Request

print(inspect.getfile(Request))
print(inspect.getsource(Request))
```

**Implementation found.** Some values are stored, while others are derived from token lists:

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

`num_prompt_tokens` describes the prompt length; `output_token_ids` exposes generated tokens; `spec_token_ids` holds speculative candidates. `num_computed_tokens` starts at zero, but its later value includes optimistically scheduled work. `num_in_flight_tokens` separately tracks scheduled work whose output has not yet been processed.

**Did the inspection answer the question?** Yes. The scheduler can express both a long prefill and a decode step as token work that has not yet been accounted for. The source also corrects an oversimplification in the chapter's introductory notes: `num_computed_tokens` is not always the number of tokens already completed on the GPU. That distinction becomes explicit in step 6 below.

### 3. Inspect RUNNING-request token-budget clipping

**Question.** How many tokens does an existing running request receive, and which limits can reduce that allocation?

**What I did.** Notebook cell 57 retrieves `Scheduler.schedule()` and filters for token-count and budget expressions:

```python
schedule_source = inspect.getsource(Scheduler.schedule)
keywords = [
    "num_new_tokens", "token_budget", "input_budget",
    "long_prefill_token_threshold", "max_model_len",
]
for line in schedule_source.splitlines():
    if any(keyword in line for keyword in keywords):
        print(line)
```

**Implementation found.** The full method, printed earlier in the notebook, provides the surrounding control flow. For running requests, the initial calculation is:

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(
    num_new_tokens, token_budget, input_budget - draft_slots
)
```

The method also caps work by the remaining model-length allowance and applies specialized alignment, encoder, and lookahead constraints. Once a request receives an allocation, its token count is saved and the budgets are deducted:

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
```

If an individual running request has zero schedulable tokens, the inspected branch advances to another request with `continue`; it does not necessarily terminate the whole round. Exhausting the shared token budget, however, ends the running-request loop.

**Did the inspection answer the question?** Yes: allocation is the remaining token demand clipped by several constraints, not a fixed number of tokens per running request. The keyword output alone omits some branch context, so it must be read alongside the complete method. No actual per-request allocations were measured in this cell.

### 4. Inspect WAITING → RUNNING admission

**Question.** When can a queued request join the active set, and where exactly does its status change?

**What I did.** Notebook cell 58 searches the same method for waiting queues, request budgets, KV allocation, `self.running.append`, and `RequestStatus.RUNNING`.

**Implementation found.** The ordinary waiting flow is entered only when there were no preemptions in the round and the scheduler is unpaused. It checks the remaining budgets and running capacity, selects a waiting queue, and examines request readiness and cached-prefix progress. For an eligible request:

```python
request_token_budget = min(token_budget, input_budget - draft_slots)
num_new_tokens = request.num_tokens - num_computed_tokens
```

Using `num_tokens` rather than only `num_prompt_tokens` also accounts for resumed requests with existing output tokens. After chunk and budget constraints, the scheduler attempts `allocate_slots(...)`. In the successful ordinary admission branch, the request is removed from its waiting queue, appended to `self.running`, and recorded in the iteration's plan before:

```python
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

**Did the inspection answer the question?** Yes. RUNNING is set during scheduling, before model execution. A request can therefore be RUNNING while still processing prefill chunks. Allocation failure in this waiting branch stops admission; the request is not promoted simply because token budget remains. The notebook identifies these conditions in source but does not measure waiting time or capture a real transition event.

### 5. Inspect KV allocation failure, victim selection, and budget rollback

**Question.** What happens if an already running request cannot obtain KV slots, and why does the code sometimes add tokens back to the budget?

**What I did.** Notebook cell 59 filters allocation and preemption logic, including `new_blocks`, `_preempt_request`, `restored`, and budget increments.

**Implementation found.** `new_blocks is None` indicates that the allocation attempt did not succeed. In the running flow, the scheduler can remove a victim and invoke `_preempt_request(...)`. Under priority scheduling, the full printed method selects a victim using the largest `(priority, arrival_time)` key. In the alternative branch, it removes the last entry from `self.running`.

The priority branch may choose a request that was already included in this round's tentative plan. That allocation must be undone:

```python
scheduled_running_reqs.remove(preempted_req)
restored = num_scheduled_tokens.pop(preempted_req_id)
token_budget += restored
input_budget += restored + draft_slots
req_to_new_blocks.pop(preempted_req_id)
```

The code also removes associated speculative or encoder planning entries where applicable. Restoring the budgets prevents canceled work from continuing to consume this iteration's capacity. The scheduler retries allocation where possible; if the current request itself is preempted or allocation remains impossible, it stops that scheduling path.

**Did the inspection answer the question?** Yes for the failure branch and its accounting. The chapter's later context dump also shows the preempted request being prepended to the waiting queue. However, these outputs do not demonstrate an actual preemption during the baseline run, nor do they establish a measured preemption cost. Queue membership and enum state must remain distinct: a preempted request can be in the waiting queue while still marked PREEMPTED.

### 6. Locate progress updates and reconcile model output

**Question.** Are computed-token counts advanced after scheduling or after execution? Where do generated tokens enter the lifecycle?

**What I did.** This step spans notebook cells 60–63. First, I searched method names containing `update_from`; the saved output identified `update_from_output` and `_update_from_kv_xfer_finished`. I then printed `Scheduler.update_from_output()`. Finally, I searched the entire scheduler class for `num_computed_tokens` and printed the surrounding lines where it increases.

```python
for name in dir(Scheduler):
    if "update_from" in name:
        print(name)

print(inspect.getsource(Scheduler.update_from_output))

scheduler_source = inspect.getsource(Scheduler)
for i, line in enumerate(scheduler_source.splitlines()):
    if "num_computed_tokens" in line:
        print(f"{i:4d}: {line}")
```

**Implementation found.** The exact context revealed `_update_after_schedule()`:

```python
for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.num_in_flight_tokens += num_scheduled_token
```

The accompanying source comments explain the order: first capture the original request progress in `SchedulerOutput` so the worker can determine input IDs, then advance scheduler accounting so later rounds can be planned. `schedule()` calls this update before returning its plan.

When model output is processed, `update_from_output()` reduces in-flight counts, reads sampled token IDs, calls `_update_request_with_output(...)`, handles stopping, and builds core outputs. It also contains correction logic for rejected speculative tokens and stale or failed work.

**Did the inspection answer the question?** Yes, and it changed the initial mental model. The increment was not located solely in post-execution processing: scheduled progress and execution reconciliation are separate operations. The notebook prints the helper call that processes generated tokens, but this chapter does not independently inspect every statement inside `_update_request_with_output()`. The printed search indices are relative to the inspected class text, not stable repository line numbers.

### 7. Inspect chunked-prefill policy and its admission gate

**Question.** What allows a long prompt to be spread across rounds, and what happens if chunking is disabled?

**What I did.** Notebook cell 64 filters the schedule method for prefill and token-budget controls. Cell 65 then prints the exact context around the waiting-request admission condition, rather than relying on keyword matches alone.

**Implementation found.** The saved source uses the configuration name `enable_chunked_prefill`; an earlier search keyword in the notebook, `chunked_prefill_enabled`, is not the exact spelling in this branch. The relevant excerpt is:

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold

if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > request_token_budget
):
    break

num_new_tokens = min(num_new_tokens, request_token_budget)
assert num_new_tokens > 0
```

The long-prefill threshold limits a candidate chunk; the available request budget can reduce it further. Disabling chunked prefill causes this admission branch to stop when its candidate work exceeds the request budget. Running requests are revisited in later rounds, allowing a partially processed prompt to continue.

**Did the inspection answer the question?** Yes: it found the threshold, the actual configuration flag, and the admission condition. It did not implement a new adaptive policy or compare chunk sizes experimentally in this chapter. Smaller chunks creating room for other requests is a scheduling mechanism; an improvement in TTFT, TPOT, or throughput would require measurements.

### Worked scheduling round: make the ordering assumption explicit

The chapter proposes a long-prefill request A with 3,000 tokens remaining, a decode request B needing one token, a waiting request C, and a 2,048-token budget. The suggested allocation of A = 2,047 and B = 1 is possible, but is not guaranteed by those quantities alone.

For this **illustrative example, not a recorded trace**, assume B is visited before A in the running list, both budgets start at 2,048, speculative draft slots are zero, KV allocations succeed, and no additional cap reduces A's chunk:

| Decision | Scheduled work | Remaining token/input budgets |
|---|---:|---:|
| Visit running decode request B | 1 token | 2,047 / 2,047 |
| Visit running prefill request A | 2,047 tokens | 0 / 0 |
| Consider waiting request C | Not admitted in this round | 0 / 0 |

If A is visited first with no smaller chunk cap, it may consume the entire 2,048-token budget and B may receive no work that round. **“RUNNING requests first” does not mean “decode requests always first.”** Running requests can include incomplete prefills. Ordering, thresholds, capacity, and allocation outcomes determine the final plan.

After the round is scheduled, progress accounting advances; execution and output reconciliation follow. Subsequent rounds revisit unfinished requests and can admit new ones when conditions allow. That changing membership is the mechanism behind continuous batching, rather than a requirement to wait for a fixed batch to finish.

### Findings and remaining runtime evidence

| Established by this chapter | Not established by its saved inspection outputs |
|---|---|
| Queue structures and configuration bindings. | Live queue sizes and per-request waiting durations. |
| Token-demand calculation and budget clipping. | Actual token allocation for every baseline iteration. |
| Successful admission and allocation-failure branches. | Whether the baseline caused preemption, and its frequency. |
| Scheduling-time progress updates and output reconciliation. | A timestamped sequence matching scheduling decisions to GPU completion. |
| Chunk threshold and chunked-prefill admission control. | Performance gains from fixed or adaptive chunk-policy changes. |


## KV Cache, PagedAttention, and Block Allocation

This section follows the corresponding chapter of [vLLM_Foundation.ipynb](https://github.com/licheng2018/vllm/blob/main/vLLM%20Setup%2C%20Request%20Inspection%2C%20and%20Baseline%20Benchmarking/vLLM_Foundation.ipynb). After understanding how the scheduler chooses token work, the next question is **how that work obtains KV storage and how the worker locates it**.

The investigation uses the saved **vLLM 0.29.0** source-inspection outputs in notebook cells **68–91**, counting all physical cells from one. Each step below preserves the notebook's progression, including searches that initially reached a wrapper or abstract method. Code excerpts are shortened for readability. The evidence establishes implementation paths; these cells do not measure live block allocation, cache-hit rates, fragmentation, or GPU memory traffic.

### 1. Inspect KVCacheManager.allocate_slots(): connect token work to capacity

**Question.** When the scheduler requests storage for N more tokens, what information does the KV manager consider, and how does it report failure?

**What I did.** Notebook cell 68 locates and prints the allocation entry point:

```python
import inspect
from vllm.v1.core.kv_cache_manager import KVCacheManager

print(inspect.getfile(KVCacheManager))
print(inspect.getsource(KVCacheManager.allocate_slots))
```

**Implementation found.** The method receives the request and `num_new_tokens`, plus optional information about newly matched prefix blocks, externally computed tokens, speculative lookahead, encoder tokens, reservations, and admission constraints. It returns `KVCacheBlocks | None`.

The calculation distinguishes already accounted-for tokens, new cache hits, and positions requiring storage. Selected expressions are:

```python
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)
```

The manager asks its coordinator how many blocks are required. Capacity is checked after accounting for reservations and admission headroom:

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```

An optional `full_sequence_must_fit` gate performs an earlier capacity check. The printed method also removes blocks no longer needed by an attention window, using a boundary that subtracts in-flight tokens so it does not rely solely on optimistic scheduling progress. After successful checks, it attaches newly computed blocks as needed and delegates new allocation to the coordinator.

**Did the inspection answer the question?** Yes. Token budget alone is insufficient: an allocation can fail because the required blocks exceed available capacity after reservations. The source also shows that an allocation attempt may perform window-related cleanup before returning `None`, so failure should not be interpreted as a guarantee that no bookkeeping changed. The notebook does not print the actual required/free-block counts for a live request.

### 2. Follow block-demand calculation through the coordinator

**Question.** Which component converts token positions into block requirements, and why is there more than one coordinator?

**What I did.** Notebook cells 69–72 search `KVCacheManager` for coordinator calls, inspect `get_kv_cache_coordinator()`, and examine `get_num_blocks_to_allocate()` on the available coordinator classes.

```python
from vllm.v1.core.kv_cache_manager import get_kv_cache_coordinator
import vllm.v1.core.kv_cache_coordinator as kv_coord

print(inspect.getsource(get_kv_cache_coordinator))
for cls in (
    kv_coord.KVCacheCoordinatorNoPrefixCache,
    kv_coord.UnitaryKVCacheCoordinator,
    kv_coord.HybridKVCacheCoordinator,
):
    print(inspect.getsource(cls.get_num_blocks_to_allocate))
```

**Implementation found.** The factory selects a coordinator according to caching configuration and the number of KV-cache groups:

| Factory condition | Selected class |
|---|---|
| Prefix caching disabled | `KVCacheCoordinatorNoPrefixCache` |
| Caching enabled and one KV-cache group | `UnitaryKVCacheCoordinator` |
| Caching enabled and multiple groups | `HybridKVCacheCoordinator` |

The coordinator's counting method iterates over its single-type managers and combines their requirements. Its inputs include the target token capacity, existing prefix-hit blocks, local/external progress, and encoder requirements. Cross-attention and other attention types can require different accounting.

**Did the inspection answer the question?** It established the delegation structure and factory conditions. A method printed through a subclass can be inherited; seeing the same implementation under several class names does not imply three separate algorithms. The notebook's next search, despite being titled “single-type KV cache managers,” still exposes coordinator-level counting in this module. The concrete allocation step below provides the clearer token-to-block calculation. These inspections do not instantiate the factory to report which coordinator the baseline actually selected.

### 3. Trace physical block allocation and request growth

**Question.** Where are new blocks obtained, how are they associated with a request, and does every token require a new block?

**What I did.** Notebook cell 73 finds `allocate_new_blocks()` implementations. Cells 74–75 inspect `BlockPool.get_new_blocks()`, `KVCacheBlock`, and pool initialization.

```python
print(inspect.getsource(kv_coord.SingleTypeKVCacheManager.allocate_new_blocks))

from vllm.v1.core.block_pool import BlockPool
print(inspect.getsource(BlockPool.get_new_blocks))
print(inspect.getsource(BlockPool.__init__))
```

**Implementation found: request capacity.** The coordinator delegates to per-type managers. In the printed single-type allocation method, the ordinary growth calculation is:

```python
req_blocks = self.req_to_blocks[request_id]
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
```

When additional capacity is needed, the method obtains blocks from the pool and extends `req_blocks`. A separate partial-prefix-hit branch can allocate a private copy-on-write block before this calculation, so “no length growth” does not universally mean “no new allocation.”

For a **simplified example** with a 16-token block size, no cache sharing, and no speculative reservations:

| Token positions requiring storage | Required blocks | Growth from two existing blocks |
|---|---:|---:|
| 0–29: 30 positions | `ceil(30 / 16) = 2` | None |
| 0–31: 32 positions | `ceil(32 / 16) = 2` | None |
| 0–32: 33 positions | `ceil(33 / 16) = 3` | One block |

This describes capacity for processed positions, not the exact instant a generated token is appended. The example block size is illustrative, not a measured configuration value from these cells.

**Implementation found: the pool.** `BlockPool.get_new_blocks()` removes entries with `free_block_queue.popleft_n(num_blocks)`. Returned blocks must have `ref_cnt == 0`, and allocation increments their reference count. When caching is enabled, a reused free block may have its previous cached mapping evicted first. An oversized direct pool request raises `ValueError`, unlike the higher-level `allocate_slots()` capacity-failure return of `None`.

`KVCacheBlock` contains a physical block ID, reference count, cache-hash metadata, and links used by the free queue. Pool initialization creates block metadata and a `FreeKVCacheBlockQueue`, and reserves a special null block that must not be freed like a normal block.

**Did the inspection answer the question?** Yes: it traces request growth to pool allocation and reference bookkeeping. These Python block objects identify slots in the KV pool; allocating one does not mean a fresh CUDA allocation is issued for each token. The chapter does not inspect initial GPU KV-tensor allocation or capture actual block IDs assigned during inference.

### 4. Inspect freeing: release references and preserve reuse opportunities

**Question.** When a request releases its blocks, when do those blocks become available, and are cached contents immediately discarded?

**What I did.** Notebook cells 76–77 print the per-type free method, discover free-related pool methods, and inspect `BlockPool.free_blocks()`.

```python
print(inspect.getsource(kv_coord.SingleTypeKVCacheManager.free))
print(inspect.getsource(BlockPool.free_blocks))
```

**Implementation found.** The single-type manager passes blocks in reverse order, releasing the tail first:

```python
self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

The pool decrements references and only queues ordinary blocks whose count reaches zero:

```python
block.ref_cnt -= 1
if block.ref_cnt == 0 and not block.is_null:
    # Classify for early or late reuse based on cache metadata.
    ...
```

The saved implementation separates blocks without cached hashes from cached blocks. Non-cached blocks are prepended for earlier reuse; cached blocks are appended for later reuse, supporting the pool's stated locality and cache-eviction behavior.

**Did the inspection answer the question?** Yes. A request releasing a reference does not necessarily make a shared block free, and a reusable cached block can retain useful cache metadata until reuse/eviction. Returning a block to this pool is also different from returning GPU memory to the CUDA allocator. The notebook does not log reference counts before and after a real completion or demonstrate how long a released prefix remains reusable.

### 5. Follow prefix-cache lookup to the concrete hash lookup

**Question.** How does vLLM discover reusable prefix KV blocks, and where is the actual cache lookup performed?

**What I did.** Notebook cells 78–82 progressively inspect coordinator lookup, search for concrete method definitions, enumerate subclasses, and finally inspect `FullAttentionManager.find_longest_cache_hit()` and `BlockPool.get_cached_block()`.

```python
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

print(inspect.getsource(FullAttentionManager.find_longest_cache_hit))
print(inspect.getsource(BlockPool.get_cached_block))
```

The intermediate subclass traversal matters: the first search encounters abstract `find_longest_cache_hit()` methods and wrappers. Enumerating `SingleTypeKVCacheManager.__subclasses__()` and following each class's `__module__` locates concrete implementations in `vllm.v1.core.single_type_kv_cache_manager`.

**Implementation found: coordinator behavior.** With prefix caching disabled, lookup returns empty hit lists and zero hit lengths. The unitary coordinator delegates to its single manager. The hybrid implementation reconciles candidate hit lengths across attention groups, reducing the candidate until their constraints agree.

**Implementation found: full-attention lookup.** After resolving hash granularity and block sizing, the full-attention implementation scans cached full blocks from the beginning:

```python
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)
hit_length = len(computed_blocks[0]) * block_size
```

The inspected version also supports a fine-grained lookup branch that probes boundaries inside the first non-full hit block. It then applies alignment and optional EAGLE-related adjustments. Consequently, a universal claim that all cache hits must end at a full physical block would be too strong for this source.

At pool level, lookup combines the block hash with each KV-cache group ID and queries `cached_block_hash_to_block`. If any required group misses, `get_cached_block()` returns `None`; otherwise it returns the matched block objects.

**Did the inspection answer the question?** Yes: it traces the path from the coordinator interface to a concrete cache map. It also shows why ordinary decode reuse and prefix caching should be distinguished: decode reuses a request's existing K/V, while prefix lookup can avoid recomputing a matching prefix. The chapter inspects lookup, but does not measure cache-hit rate, inspect the full hash-construction path, or run a repeated-prefix performance comparison.

### 6. Follow block IDs into ModelRunner and the slot-mapping kernel

**Question.** How does a scheduler-side allocation become a physical KV location that GPU work can use?

**What I did.** Notebook cells 83–91 follow this path in several stages:

| Notebook cells | Inspection | What the saved source reveals |
|---|---|---|
| 83–84 | Scheduler output construction and request-data classes | New requests carry `block_ids`; cached-request updates carry `new_block_ids` and resumed-request information. |
| 85–86 | `GPUModelRunner` keyword search and surrounding code | Build cached request state, extend mappings for growth, replace mappings on resume where needed, and update batch block tables. |
| 87–88 | `compute_slot_mapping()` in the input-batch module | Initially finds `MultiGroupBlockTable`, a wrapper delegating to per-group tables. |
| 89 | Follow the defining module | Finds `BlockTable.compute_slot_mapping()` in `vllm.v1.worker.block_table`. |
| 90–91 | Inspect the callable kernel object, then its class | Direct object inspection fails; class inspection exposes `ComputeSlotMappingKernel` and its Triton kernel. |

**Implementation found: worker state.** `NewRequestData.from_request(...)` combines request fields with block IDs. The runner uses these values to construct `CachedRequestState`. Existing requests can extend their per-group block lists; a preempted/resumed request can replace its mapping. The inspected runner also commits the block table and invokes slot mapping with request boundaries and token positions.

This is metadata propagation, not a transfer of all cached K/V values inside `SchedulerOutput`. The block table tells execution where the relevant storage resides.

**Implementation found: wrapper versus kernel.** The multi-group wrapper calls each table's `compute_slot_mapping()`. The per-group table then dispatches `_COMPUTE_SLOT_MAPPING_KERNEL` with `query_start_loc`, `positions`, the GPU block table, block sizes, output slot storage, and context-parallel configuration. It has a separate no-slot-mapping path for state-based groups such as Mamba/GDN.

The notebook's first attempt to inspect the callable fails because the object is a `ComputeSlotMappingKernel` instance. The successful follow-up is:

```python
import vllm.v1.worker.block_table as block_table_module

kernel = block_table_module._COMPUTE_SLOT_MAPPING_KERNEL
KernelClass = type(kernel)
print(inspect.getfile(KernelClass))
print(inspect.getsource(KernelClass))
```

The printed Triton implementation finds each request's token range, loads positions, resolves block-table entries, and writes slot IDs. The final address calculation includes:

```python
slot_offsets = local_block_offsets % block_size
slot_ids = block_numbers * block_size + slot_offsets
```

It also handles context-parallel locality and padding. In a **simplified single-rank example** with equal 16-token allocation and KV block sizes, the mapping can be expressed as:

```text
logical_block = token_position // 16
physical_block = request_block_table[logical_block]
slot = physical_block * 16 + token_position % 16
```

For a hypothetical table `[7, 21, 4]`, token position 32 maps to logical block 2, physical block 4, and slot 64. Physical adjacency between blocks 7, 21, and 4 is unnecessary. This is an illustrative calculation, not a slot value printed by the notebook.

**Did the inspection answer the question?** Yes for the route from allocation metadata to token-to-slot mapping. It does not inspect the full attention computation that reads those K/V values, dump real GPU mappings, or benchmark the attention kernel. The block table and mapping explain an essential part of paged KV management; the slot-mapping kernel itself is not the complete PagedAttention algorithm.

### What the allocation walkthrough establishes

The source-level chain is:

```text
Scheduler chooses token work
    → KVCacheManager checks demand and capacity
    → Coordinator delegates to per-type managers
    → BlockPool supplies reusable physical block IDs
    → SchedulerOutput carries request/block metadata
    → ModelRunner updates request state and block tables
    → Slot-mapping kernel resolves token positions to KV slots
```

Prefix-cache lookup can reuse existing computed blocks along this path. Completion or preemption can release references back to the pool, subject to sharing and execution-lifetime constraints. Neither operation requires a request's logical sequence to occupy one contiguous physical range.

| Established by saved source inspection | Still requires runtime evidence |
|---|---|
| Capacity checks, reservations, and the `None` failure path. | Actual required/free-block counts per scheduling round. |
| Block-growth arithmetic and reference-counted pool allocation. | Observed allocation sizes, fragmentation, and memory utilization. |
| Free-queue ordering and cache-aware block reuse. | Real reference-count transitions and cache eviction frequency. |
| Prefix lookup across hashes and KV-cache groups. | Cache-hit rates and saved prefill work for benchmark workloads. |
| Block-ID propagation and token-to-slot computation. | Concrete GPU slot mappings and attention memory-access measurements. |
| The scheduling link between KV capacity and admission/preemption. | Actual preemption frequency, recomputation cost, and latency impact. |


## Scheduler Instrumentation and Benchmarking

Scheduler instrumentation records per-step state in `scheduler_trace.jsonl`, supported by a trace script. The baseline benchmark covers four measurement areas:

| Measurement | Focus |
|---|---|
| Time to first token (TTFT) | Initial response latency. |
| Time per output token (TPOT) | Token generation latency after the first token. |
| Throughput | Serving output rate. |
| KV-cache pressure | Cache usage and memory constraints during serving. |

The benchmark work produced a baseline table and plots. Numerical results and trace artifacts are not included on this page yet.

## Chunked Prefill Policy Experiments

The scheduler modification work started with a fixed chunked prefill policy and then extended to an adaptive policy. The outputs include a scheduler patch and a correctness test. This page documents the implementation scope; measured performance gains and correctness-test results are not yet reported here.

## Skills Demonstrated

- vLLM environment setup and offline/server inference execution.
- Source-code tracing across the request lifecycle.
- Scheduler analysis: waiting/running states, token budgets, and continuous batching.
- KV-cache management, PagedAttention, and preemption analysis.
- Scheduler instrumentation and structured trace collection.
- Inference benchmarking with TTFT, TPOT, throughput, and KV-cache pressure.
- Fixed and adaptive chunked prefill policy modification and correctness testing.

[Back to Home](../index.md)
