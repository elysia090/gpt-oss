# Sera Transfer CLI

The conversion tool lives in `gpt_oss.tools.sera_transfer` and can be invoked with the standard Python module runner:

```bash
python -m gpt_oss.tools.sera_transfer --source /path/to/checkpoint --output /path/to/output --r 512 --rv 12 --topL 12
```

> **Prerequisite**
> Install the official safetensors wheel before running the converter, for
> example with `pip install safetensors` or `pip install "gpt-oss[torch]"`.
> The upstream package now handles binary `model.safetensors` checkpoints;
> the repository's JSON-only stub remains available strictly for tests.

The source directory must contain both `model.safetensors` and `config.json`.
When those files are missing from the immediate `SOURCE/` root the converter
automatically probes `SOURCE/original/` and `SOURCE/original/model/` before
failing.  If your layout differs, pass `--original-subdir` with a custom
relative path.  For example, to target `SOURCE/checkpoints/final/`:

```bash
python -m gpt_oss.tools.sera_transfer \
  --source /path/to/checkpoint \
  --original-subdir checkpoints/final \
  --output /path/to/output
```

To download the artefacts from the Hub you can use the `hf` CLI:

```bash
hf download my-org/my-model original/config.json original/model.safetensors --local-dir /path/to/checkpoint
```

The command produces a `sera_manifest.bin` file alongside a populated
`output/arrays/` directory that matches the layout described in
`docs/specs/Sera-Transfer.txt`.

During testing the helper accepts smaller values for `--r`, `--rv` and `--topL`;
the defaults clamp to the model's hidden size to guarantee that matrix
multiplications remain well defined.
