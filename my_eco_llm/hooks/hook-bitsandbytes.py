import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


def _is_selected_binary(name: str, cuda_tag: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("libbitsandbytes_cpu"):
        return True
    if cuda_tag and lowered.startswith(f"libbitsandbytes_cuda{cuda_tag}"):
        return True
    return False


cuda_tag = os.environ.get("PYI_BNB_CUDA_TAG", "").strip()

binaries = []
for source, destination in collect_dynamic_libs("bitsandbytes"):
    if _is_selected_binary(Path(source).name, cuda_tag):
        binaries.append((source, destination))

# bitsandbytes triton kernels require source files at runtime.
module_collection_mode = "pyz+py"
