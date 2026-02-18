import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


hiddenimports = []

spec = importlib.util.find_spec("unsloth_zoo")
if spec and spec.submodule_search_locations:
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    for py_file in package_dir.rglob("*.py"):
        relative = py_file.relative_to(package_dir)
        if py_file.name == "__init__.py":
            if len(relative.parts) == 1:
                module_name = "unsloth_zoo"
            else:
                module_name = "unsloth_zoo." + ".".join(relative.parts[:-1])
        else:
            module_name = "unsloth_zoo." + ".".join(relative.with_suffix("").parts)
        hiddenimports.append(module_name)

hiddenimports = sorted(set(hiddenimports))
# Keep source files so inspect.getsource() used by unsloth_zoo RL patches works in frozen runtime.
datas = collect_data_files("unsloth_zoo", include_py_files=True)
