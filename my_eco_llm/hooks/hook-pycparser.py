import importlib.metadata


def _pycparser_major_minor() -> tuple[int, int]:
    try:
        raw = importlib.metadata.version("pycparser")
    except importlib.metadata.PackageNotFoundError:
        return (0, 0)
    parts = raw.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


# pycparser >= 3 switched parser implementation and no longer uses generated
# lextab/yacctab modules.
if _pycparser_major_minor() < (3, 0):
    hiddenimports = ["pycparser.lextab", "pycparser.yacctab"]
else:
    hiddenimports = []
