# Sera Transfer CLI

The conversion tool lives in `gpt_oss.tools.sera_transfer` and can be invoked with the standard Python module runner:

```bash
python -m gpt_oss.tools.sera_transfer --source /path/to/checkpoint --output /path/to/output --r 512 --rv 12 --topL 12
```

The source directory must contain both `model.safetensors` and `config.json`.  The command produces a `sera_manifest.bin` file alongside a populated `output/arrays/` directory that matches the layout described in `docs/specs/Sera-Transfer.txt`.

During testing the helper accepts smaller values for `--r`, `--rv` and `--topL`; the defaults clamp to the model's hidden size to guarantee that matrix multiplications remain well defined.
