from __future__ import annotations

from pathlib import Path

import torch
import torchao


def patch_torch_inductor_codecache() -> bool:
    target = Path(torch.__file__).resolve().parent / "_inductor" / "codecache.py"
    text = target.read_text(encoding="utf-8")

    if 'os.path.exists("/usr/lib64/libgomp.so.1")' not in text and 'cdll.LoadLibrary("/usr/lib64/libgomp.so.1")' not in text:
        return False

    patched = text.replace(
        '            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):\n'
        "                # hacky workaround for fbcode/buck\n"
        "                global _libgomp\n"
        '                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")\n',
        '            linux_libgomp = os.path.join(os.sep, "usr", "lib64", "libgomp.so.1")\n'
        '            if "gomp" in str(e) and os.path.exists(linux_libgomp):\n'
        "                # hacky workaround for fbcode/buck\n"
        "                global _libgomp\n"
        "                _libgomp = cdll.LoadLibrary(linux_libgomp)\n",
    )
    if patched != text:
        target.write_text(patched, encoding="utf-8")
        return True
    return False


def patch_torchao_quant_api_doc_examples() -> bool:
    target = Path(torchao.__file__).resolve().parent / "quantization" / "quant_api.py"
    text = target.read_text(encoding="utf-8")
    patched = text.replace(
        "re:language\\.layers\\..+\\.q_proj.weight",
        "re:language\\\\.layers\\\\..+\\\\.q_proj.weight",
    ).replace(
        "re:language\\.layers\\..+\\.q_proj",
        "re:language\\\\.layers\\\\..+\\\\.q_proj",
    )
    if patched != text:
        target.write_text(patched, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed_codecache = patch_torch_inductor_codecache()
    changed_torchao = patch_torchao_quant_api_doc_examples()
    print(
        "{"
        f'"patched_torch_inductor_codecache": {int(changed_codecache)}, '
        f'"patched_torchao_quant_api_doc_examples": {int(changed_torchao)}'
        "}"
    )


if __name__ == "__main__":
    main()
