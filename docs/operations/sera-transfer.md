# Sera Transfer CLI

The Sera Transfer CLI converts Hugging Face checkpoints into the manifest format consumed by the Sera runtime. It validates the source model, materializes the binary arrays, and emits a `sera_manifest.bin` file alongside the tensors required by `gpt-oss-sera-chat` and related tooling.

Use this guide when you need to:

- Convert a freshly downloaded checkpoint into the Sera layout.
- Automate conversions in CI/CD pipelines.
- Inspect or rebuild manifests that ship with pre-generated artifacts.

## Prerequisites

Before running the converter make sure the following tools are installed:

- Python 3.12 or newer.
- The [`safetensors`](https://pypi.org/project/safetensors/) wheel for reading binary checkpoints.
- Optional: the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) if you plan to download artifacts from the Hub as part of the workflow.

Install the conversion entry point from the repository or PyPI distribution:

```bash
pip install gpt-oss
# or
pip install "gpt-oss[torch]"
```

The CLI lives in `gpt_oss.tools.sera_transfer` and is invokable with Python's module runner.

## Basic usage

To convert a checkpoint stored locally:

```bash
python -m gpt_oss.tools.sera_transfer \
  --source /path/to/checkpoint \
  --output /path/to/output \
  --r 512 --rv 12 --topL 12
```

By default the command expects `model.safetensors` and `config.json` to be present at the root of `--source`. When they are nested under an `original/` directory the helper discovers them automatically. Set `--original-subdir` to point at a custom relative path if your layout differs.

## Downloading from Hugging Face

Use the Hugging Face CLI to fetch the original safetensors payload and configuration:

```bash
hf download my-org/my-model original/config.json original/model.safetensors \
  --local-dir /path/to/checkpoint
```

After the download completes, run the converter against `/path/to/checkpoint`.

## Output structure

The converter creates the following artifacts:

- `sera_manifest.bin` — manifest consumed by the Sera runtime.
- `arrays/` — directory containing MXFP4 shard data and auxiliary tensors.
- `logs/` (optional) — diagnostic logs if you pass `--debug`.

Use the manifest with the `gpt-oss-sera-chat` CLI or bundle it alongside your own Sera-based tooling.

## Adjusting tensor dimensions

For test environments or custom research flows you can clamp tensor dimensions without rebuilding the entire checkpoint. Pass the following flags to override the defaults that are inferred from the model config:

- `--r` — hidden size.
- `--rv` — rotary value dimension.
- `--topL` — number of experts loaded per token.

Values lower than the model defaults produce smaller arrays but are not compatible with the published 20B and 120B checkpoints. Use them only with fixtures or synthetic data.

## Troubleshooting

The CLI validates metadata and shapes before writing the manifest. If validation fails, the exception message includes remediation hints and a pointer back to this document. Common causes include:

- Missing or incomplete `model.safetensors` files.
- Incorrect tensor shapes when experimenting with synthetic inputs.
- Typographical errors in `--original-subdir` values.

Re-run the command with `--debug` to emit verbose logs and inspect intermediate tensor metadata.
