<img alt="gpt-oss-120" src="./docs/gpt-oss.svg">
<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides</strong></a> ·
  <a href="https://arxiv.org/abs/2508.10925"><strong>Model card</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss/"><strong>OpenAI blog</strong></a>
</p>
<p align="center">
  <strong>Download <a href="https://huggingface.co/openai/gpt-oss-120b">gpt-oss-120b</a> and <a href="https://huggingface.co/openai/gpt-oss-20b">gpt-oss-20b</a> on Hugging Face</strong>
</p>

<br>

Welcome to the gpt-oss series, [OpenAI's open-weight models](https://openai.com/open-models/) designed for powerful reasoning, agentic tasks, and versatile developer use cases.

We're releasing two flavors of these open models:

- `gpt-oss-120b` — for production, general purpose, high reasoning use cases that fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters)
- `gpt-oss-20b` — for lower latency, and local or specialized use cases (21B parameters with 3.6B active parameters)

Both models were trained using our [harmony response format][harmony] and should only be used with this format; otherwise, they will not work correctly.

## Table of Contents
- [Overview](#overview)
- [Model lineup](#model-lineup)
- [Highlights](#highlights)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Install the packages](#install-the-packages)
  - [Download model weights](#download-model-weights)
- [Running the models](#running-the-models)
  - [Transformers](#transformers)
  - [vLLM](#vllm)
  - [Reference runtimes](#reference-runtimes)
- [Harmony format and native tools](#harmony-format-and-native-tools)
  - [Harmony chat format](#harmony-chat-format)
  - [Browser tool](#browser-tool)
  - [Python tool](#python-tool)
  - [Apply Patch tool](#apply-patch-tool)
  - [Sera terminal experience](#sera-terminal-experience)
- [Repository guide](#repository-guide)
  - [Layout](#layout)
  - [Local development](#local-development)
- [Additional clients](#additional-clients)
- [Other details](#other-details)
  - [Precision format](#precision-format)
  - [Recommended sampling parameters](#recommended-sampling-parameters)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

The gpt-oss project provides production-grade reference implementations, tooling, and documentation to help developers evaluate, deploy, and extend the gpt-oss-120b and gpt-oss-20b models. Each component in this repository is designed to demonstrate best practices for running the models reliably across a range of environments—from single GPU developer rigs to multi-GPU inference clusters.

## Model lineup

| Model | Best for | Hardware profile |
| --- | --- | --- |
| `gpt-oss-120b` | Highest-quality, general-purpose reasoning | Fits on a single 80GB GPU (H100 or MI300X) using MXFP4 quantization |
| `gpt-oss-20b` | Latency-sensitive, local, and specialized deployments | Runs within 16GB of memory |

Both models expose the full chain-of-thought via the Harmony format to facilitate debugging, agentic orchestration, and advanced tooling integrations.

## Highlights

- **Permissive Apache 2.0 license:** Build freely without copyleft restrictions or patent risk—ideal for experimentation, customization, and commercial deployment.
- **Configurable reasoning effort:** Easily adjust the reasoning effort (low, medium, high) based on your specific use case and latency needs.
- **Full chain-of-thought:** Provides complete access to the model's reasoning process, facilitating easier debugging and greater trust in outputs. This information is not intended to be shown to end users.
- **Fine-tunable:** Fully customize models to your specific use case through parameter fine-tuning.
- **Agentic capabilities:** Use the models' native capabilities for function calling, [web browsing](#browser-tool), [Python code execution](#python-tool), and Structured Outputs.
- **MXFP4 quantization:** The models were post-trained with MXFP4 quantization of the MoE weights, making `gpt-oss-120b` run on a single 80GB GPU (like NVIDIA H100 or AMD MI300X) and the `gpt-oss-20b` model run within 16GB of memory. All evals were performed with the same MXFP4 quantization.

## Getting started

### Requirements

- Python 3.12
- On macOS: install the Xcode CLI tools (`xcode-select --install`).
- On Linux: CUDA-capable hardware is required for the reference GPU runtimes.
- On Windows: the reference implementations have not been tested. Use solutions like Ollama for local evaluation.

### Install the packages

Install the package from [PyPI](https://pypi.org/project/gpt-oss/) based on the components you plan to use:

```shell
# tooling only
pip install gpt-oss
# PyTorch reference runtime
pip install gpt-oss[torch]
# Triton reference runtime
pip install gpt-oss[triton]
```

To develop locally or experiment with the Metal runtime, clone the repository and install in editable mode:

```shell
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
```

### Download model weights

You can download the model weights from the [Hugging Face Hub](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) using the CLI:

```shell
# gpt-oss-120b
hf download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/

# gpt-oss-20b
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

Check out our [awesome list](./awesome-gpt-oss.md) for a broader collection of gpt-oss resources and inference partners.

## Running the models

### Transformers

You can use `gpt-oss-120b` and `gpt-oss-20b` with the Transformers library. If you use Transformers' chat template, it will automatically apply the [harmony response format][harmony]. If you use `model.generate` directly, you need to apply the harmony format manually using the chat template or use our [`openai-harmony`][harmony] package.

```python
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

[Learn more about how to use gpt-oss with Transformers.](https://cookbook.openai.com/articles/gpt-oss/run-transformers)

### vLLM

vLLM recommends using [`uv`](https://docs.astral.sh/uv/) for Python dependency management. You can use vLLM to spin up an OpenAI-compatible web server. The following command will automatically download the model and start the server.

```bash
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b
```

[Learn more about how to use gpt-oss with vLLM.](https://cookbook.openai.com/articles/gpt-oss/run-vllm)

Offline Serve Code:
- run this code after installing proper libraries as described, while additionally installing this:
- `uv pip install openai-harmony`
```python
# source .oss/bin/activate

import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

from vllm import LLM, SamplingParams
import os

# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)

prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# --- 2) Run vLLM with prefill ---
llm = LLM(
    model="openai/gpt-oss-20b",
    trust_remote_code=True,
    gpu_memory_utilization = 0.95,
    max_num_batched_tokens=4096,
    max_model_len=5000,
    tensor_parallel_size=1
)

sampling = SamplingParams(
    max_tokens=128,
    temperature=1,
    stop_token_ids=stop_token_ids,
)

outputs = llm.generate(
    prompt_token_ids=[prefill_ids],   # batch of size 1
    sampling_params=sampling,
)

# vLLM gives you both text and token IDs
gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)

# --- 3) Parse the completion token IDs back into structured Harmony messages ---
entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

# 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
for message in entries:
    print(f"{json.dumps(message.to_dict())}")
```

### Reference runtimes

#### PyTorch

We include an educational PyTorch implementation in [src/gpt_oss/torch/model.py](src/gpt_oss/torch/model.py). It mirrors the model architecture with minimal optimizations and supports tensor parallelism in the MoE layers so that the large model can run on 4×H100 or 2×H200 GPUs. The runtime upcasts weights to BF16.

```shell
pip install -e ".[torch]"

# On 4×H100
torchrun --nproc-per-node=4 -m gpt_oss.cli.generate_cli gpt-oss-120b/original/
```

#### Triton (single GPU)

The optimized Triton runtime relies on [the Triton MoE kernel](https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels) with MXFP4 support and reduced-memory attention kernels. Install Triton from source alongside the gpt-oss extras to run `gpt-oss-120b` on a single 80GB GPU.

```shell
# Install Triton from source
git clone https://github.com/triton-lang/triton
cd triton/
pip install -r python/requirements.txt
pip install -e . --verbose --no-build-isolation
pip install -e python/triton_kernels

# Install the gpt-oss Triton runtime
pip install -e ".[triton]"

# On 1×H100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.cli.generate_cli --backend triton gpt-oss-120b/original/
```

Turn on the expandable allocator if you encounter `torch.OutOfMemoryError` when loading weights.

#### Metal

The Metal reference runtime targets Apple Silicon and is accurate to the PyTorch implementation. Install with the Metal extra to trigger compilation:

```shell
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
```

Before running inference, convert the SafeTensor weights:

```shell
python src/gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>
```

Or download pre-converted weights:

```shell
hf download openai/gpt-oss-120b --include "metal/*" --local-dir gpt-oss-120b/metal/
hf download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/
```

Test the runtime with:

```shell
python src/gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "why did the chicken cross the road?"
```

## Harmony format and native tools

### Harmony chat format

The gpt-oss models expect prompts encoded with the Harmony chat format. The [`openai-harmony` library][harmony] provides helpers to render structured conversations, reason about stop tokens, and parse tool calls emitted by the model. Refer to the [Harmony guide](https://cookbook.openai.com/articles/openai-harmony) for the full specification.

### Browser tool

The browser tool exposes a crawl-and-cite workflow where the model scrolls through content, caches results, and emits citations in responses. The implementation lives in [src/gpt_oss/tools/browser](src/gpt_oss/tools/browser). Always create a fresh browser instance for each request so cached state remains isolated.

### Python tool

The Python tool allows the model to execute code snippets inside a sandboxed environment as part of its reasoning loop. The stateless reference implementation overrides the default Harmony description, so remember to embed the tool definition in the system message.

```python
import datetime
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import SystemContent, Message, Conversation, Role, load_harmony_encoding, HarmonyEncodingName

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
python_tool = PythonTool()

system_message_content = SystemContent.new().with_conversation_start_date(
    datetime.datetime.now().strftime("%Y-%m-%d")
)

system_message_content = system_message_content.with_tools(python_tool.tool_config)

system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
messages = [system_message, Message.from_role_and_content(Role.USER, "What's the square root of 9001?")]
conversation = Conversation.from_messages(messages)

token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
# ... run inference ...
messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
if messages[-1].recipient == "python":
    response_messages = await python_tool.process(messages[-1])
    messages.extend(response_messages)
```

> [!WARNING]
> The Python tool runs inside a permissive Docker container and is intended only as a reference. Harden the sandbox and enforce your own security controls before deploying in production.

### Apply Patch tool

`apply_patch` can be used to create, update, or delete files locally. It serves as a minimal example of file-editing workflows driven by the model.

### Sera terminal experience

The [Sera runtime](docs/sera.md) ships with a rich terminal client exposed through the `gpt-oss-sera-chat` console script. It downloads checkpoints, prepares the Sera manifest, and launches a split-pane interface with conversation, tool status, and diagnostics panes. Consult the CLI help (`gpt-oss-sera-chat --help`) for advanced options and review the [Sera Transfer guide](docs/operations/sera-transfer.md) if you need to materialize assets manually.

## Repository guide

### Layout

```
.
├── src/gpt_oss/         # Primary Python package sources
│   ├── api/             # FastAPI-powered surfaces (e.g. Responses API)
│   ├── cli/             # Command-line entry points
│   ├── inference/       # Backend-specific inference implementations
│   └── tools/           # Reference tool implementations
├── tools/mcp_server/    # MCP-compatible wrappers around the reference tools
├── docs/                # Project documentation and diagrams
├── examples/            # Sample scripts demonstrating API usage
├── tests/               # Automated tests and shared fixtures
└── tests/data/          # Test resource files consumed by the suite
```

The `src/` layout keeps the importable package separate from auxiliary tooling, while the `tools/` and `tests/` directories group developer utilities and quality checks in predictable locations.

### Local development

Install the development extras (`pip install -e .[dev]`) to pull in linting and test dependencies. The repository relies on [pytest](https://docs.pytest.org/), [ruff](https://docs.astral.sh/ruff/), and [mypy](https://mypy-lang.org/) for validation. Run the following commands before sending changes:

```shell
pytest
ruff check src tests
mypy src
```

## Additional clients

The models run well on popular community runtimes:

- **Ollama**

  ```bash
  # gpt-oss-20b
  ollama pull gpt-oss:20b
  ollama run gpt-oss:20b

  # gpt-oss-120b
  ollama pull gpt-oss:120b
  ollama run gpt-oss:120b
  ```

  [Learn more about how to use gpt-oss with Ollama.](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)

- **LM Studio**

  ```bash
  # gpt-oss-20b
  lms get openai/gpt-oss-20b
  # gpt-oss-120b
  lms get openai/gpt-oss-120b
  ```

  [Learn more about how to use gpt-oss with LM Studio.](https://cookbook.openai.com/articles/gpt-oss/run-locally-lmstudio)

- **Awesome gpt-oss** — community adapters, tools, and integrations curated in [awesome-gpt-oss.md](./awesome-gpt-oss.md).

## Other details

### Precision format

We released the models with native quantization support. Specifically, we use [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) for the linear projection weights in the MoE layer. We store the MoE tensor in two parts:

- `tensor.blocks` stores the actual fp4 values. We pack every two values in one `uint8` value.
- `tensor.scales` stores the block scale. The block scaling is done among the last dimension for all MXFP4 tensors.

All other tensors are stored in BF16. We recommend using BF16 as the activation precision for the model.

### Recommended sampling parameters

We recommend sampling with `temperature=1.0` and `top_p=1.0`.

## Contributing

The reference implementations in this repository are meant as a starting point and inspiration. Outside of bug fixes we do not intend to accept new feature contributions. If you build implementations based on this code such as new tool implementations you are welcome to contribute them to the [`awesome-gpt-oss.md`](./awesome-gpt-oss.md) file.

[harmony]: https://github.com/openai/harmony

## Citation

```bibtex
@misc{openai2025gptoss120bgptoss20bmodel,
      title={gpt-oss-120b & gpt-oss-20b Model Card},
      author={OpenAI},
      year={2025},
      eprint={2508.10925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10925},
}
```
