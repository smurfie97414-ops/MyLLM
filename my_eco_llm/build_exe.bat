@echo off
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
echo [build] Working directory: %cd%

tasklist /fi "imagename eq eco_train.exe" | find /i "eco_train.exe" >nul
if not errorlevel 1 (
  echo [build][error] eco_train.exe is currently running. Stop training before rebuilding.
  exit /b 1
)

set "PYTHON=py -3.13"
if exist "C:\Python313\python.exe" set "PYTHON=C:\Python313\python.exe"
set "REQ_FILE=requirements-py313-cu128.txt"
set "HOOKS_DIR=%cd%\hooks"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
set "VCVARS64=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"

if exist "%VSDEVCMD%" (
  echo [build] Loading VS2026 environment via VsDevCmd...
  call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul 2>&1
) else if exist "%VCVARS64%" (
  echo [build] Loading VS2026 environment via vcvars64...
  call "%VCVARS64%" >nul 2>&1
) else (
  echo [build][warn] VS2026 dev environment script not found.
)

set "TORCH_NVCC_FLAGS=--allow-unsupported-compiler"
set "CUDAFLAGS=--allow-unsupported-compiler"
set "CMAKE_CUDA_FLAGS=--allow-unsupported-compiler"
set "UNSLOTH_COMPILE_LOCATION=unsloth_compiled_cache"
if exist "unsloth_compiled_cache" (
  rmdir /s /q "unsloth_compiled_cache"
)
mkdir "unsloth_compiled_cache" >nul 2>&1
if not exist "unsloth_compiled_cache\__init__.py" (
  type nul > "unsloth_compiled_cache\__init__.py"
)

set "LIBTORCH_DIR="
if exist "%cd%\libtorch\lib" set "LIBTORCH_DIR=%cd%\libtorch"
if "%LIBTORCH_DIR%"=="" if exist "%cd%\..\libtorch\lib" set "LIBTORCH_DIR=%cd%\..\libtorch"
if not "%LIBTORCH_DIR%"=="" (
  echo [build] Using local libtorch from %LIBTORCH_DIR%
  set "LIBTORCH=%LIBTORCH_DIR%"
  set "PATH=%LIBTORCH_DIR%\lib;%PATH%"
  set "LIB=%LIBTORCH_DIR%\lib;%LIB%"
  set "INCLUDE=%LIBTORCH_DIR%\include;%LIBTORCH_DIR%\include\torch\csrc\api\include;%INCLUDE%"
)

echo [build] Ensuring packaging dependency...
echo [build] Python command: %PYTHON%
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
  echo [build][warn] pip self-upgrade failed; continuing with current pip.
)
if not exist "%REQ_FILE%" (
  echo [build][error] missing dependency lock file: %REQ_FILE%
  exit /b 1
)
echo [build] Installing pinned dependencies from %REQ_FILE% ...
%PYTHON% -m pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu128 -r "%REQ_FILE%"
if errorlevel 1 goto :fail
echo [build] Verifying runtime imports and required symbols...
%PYTHON% scripts\verify_runtime.py
if errorlevel 1 goto :fail
echo [build] Applying third-party packaging patches...
%PYTHON% scripts\patch_third_party_for_packaging.py
if errorlevel 1 goto :fail
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import torch; print((torch.version.cuda or '').replace('.', ''))"`) do set "PYI_BNB_CUDA_TAG=%%I"
if "%PYI_BNB_CUDA_TAG%"=="" set "PYI_BNB_CUDA_TAG=128"
echo [build] bitsandbytes target CUDA tag: %PYI_BNB_CUDA_TAG%

for /f "usebackq delims=" %%I in (`%PYTHON% -c "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['include']))"`) do set "PY_INCLUDE_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import pathlib,sys; print(pathlib.Path(sys.base_exec_prefix) / 'libs')"`) do set "PY_LIB_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import torch,pathlib; print(pathlib.Path(torch.__file__).resolve().parent)"`) do set "TORCH_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import transformers,pathlib; print(pathlib.Path(transformers.__file__).resolve().parent)"`) do set "TRANSFORMERS_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import accelerate,pathlib; print(pathlib.Path(accelerate.__file__).resolve().parent)"`) do set "ACCELERATE_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import peft,pathlib; print(pathlib.Path(peft.__file__).resolve().parent)"`) do set "PEFT_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import trl,pathlib; print(pathlib.Path(trl.__file__).resolve().parent)"`) do set "TRL_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import triton,pathlib; print(pathlib.Path(triton.__file__).resolve().parent)"`) do set "TRITON_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import torchao,pathlib; print(pathlib.Path(torchao.__file__).resolve().parent)"`) do set "TORCHAO_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import wandb,pathlib; print(pathlib.Path(wandb.__file__).resolve().parent)"`) do set "WANDB_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import wandb_workspaces,pathlib; print(pathlib.Path(wandb_workspaces.__file__).resolve().parent)"`) do set "WANDB_WORKSPACES_PKG_DIR=%%I"
set "TORCH_INCLUDE_DIR=%TORCH_PKG_DIR%\include"
set "TORCH_LIB_DIR=%TORCH_PKG_DIR%\lib"
echo [build] Python include dir: %PY_INCLUDE_DIR%
echo [build] Python libs dir: %PY_LIB_DIR%
echo [build] Torch package dir: %TORCH_PKG_DIR%
echo [build] Transformers package dir: %TRANSFORMERS_PKG_DIR%
echo [build] Accelerate package dir: %ACCELERATE_PKG_DIR%
echo [build] PEFT package dir: %PEFT_PKG_DIR%
echo [build] TRL package dir: %TRL_PKG_DIR%
echo [build] Triton package dir: %TRITON_PKG_DIR%
echo [build] TorchAO package dir: %TORCHAO_PKG_DIR%
echo [build] wandb package dir: %WANDB_PKG_DIR%
echo [build] wandb_workspaces package dir: %WANDB_WORKSPACES_PKG_DIR%
echo [build] Torch include dir: %TORCH_INCLUDE_DIR%
echo [build] Torch lib dir: %TORCH_LIB_DIR%

if exist "build" (
  rmdir /s /q "build"
)
if exist "dist\eco_train" (
  rmdir /s /q "dist\eco_train"
)
if exist "dist\eco_train.exe" (
  del /f /q "dist\eco_train.exe"
)
if exist "eco_train.exe" (
  del /f /q "eco_train.exe"
)
if exist "_internal" (
  rmdir /s /q "_internal"
)
if exist "eco_train.spec" (
  del /f /q "eco_train.spec"
)

echo [build] Building eco_train.exe ...
%PYTHON% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --console ^
  --name eco_train ^
  --additional-hooks-dir "%HOOKS_DIR%" ^
  --exclude-module transformers ^
  --exclude-module accelerate ^
  --exclude-module peft ^
  --exclude-module trl ^
  --exclude-module triton ^
  --exclude-module torchao ^
  --exclude-module wandb ^
  --exclude-module wandb_workspaces ^
  --exclude-module pyglet ^
  --paths . ^
  --add-data "model\triton_kernel_src.py;model" ^
  --add-data "model\sigma_kernel_src.py;model" ^
  --add-data "training\optimizer_gn_kernel_src.py;training" ^
  --add-data "training\sigma_trainer.py;training" ^
  --add-data "evolution\meta_loop.py;evolution" ^
  --add-data "%PY_INCLUDE_DIR%;Include" ^
  --add-data "%PY_LIB_DIR%;libs" ^
  --add-data "%TORCH_INCLUDE_DIR%;torch\include" ^
  --add-data "%TORCH_LIB_DIR%;torch\lib" ^
  --add-data "%TRANSFORMERS_PKG_DIR%;transformers" ^
  --add-data "%ACCELERATE_PKG_DIR%;accelerate" ^
  --add-data "%PEFT_PKG_DIR%;peft" ^
  --add-data "%TRL_PKG_DIR%;trl" ^
  --add-data "%TRITON_PKG_DIR%;triton" ^
  --add-data "%TORCHAO_PKG_DIR%;torchao" ^
  --add-data "%WANDB_PKG_DIR%;wandb" ^
  --add-data "%WANDB_WORKSPACES_PKG_DIR%;wandb_workspaces" ^
  --collect-data tiktoken ^
  --copy-metadata transformers ^
  --copy-metadata accelerate ^
  --copy-metadata peft ^
  --copy-metadata trl ^
  --copy-metadata torchao ^
  --copy-metadata wandb ^
  --copy-metadata wandb-workspaces ^
  --copy-metadata regex ^
  --copy-metadata tokenizers ^
  --copy-metadata safetensors ^
  --copy-metadata huggingface-hub ^
  --copy-metadata tqdm ^
  --copy-metadata requests ^
  --copy-metadata packaging ^
  --copy-metadata filelock ^
  --copy-metadata numpy ^
  --copy-metadata pyyaml ^
  --copy-metadata datasets ^
  --collect-data unsloth ^
  --collect-data unsloth_zoo ^
  --collect-data google.protobuf ^
  --collect-all tokenizers ^
  --collect-all xformers ^
  --copy-metadata unsloth ^
  --copy-metadata unsloth_zoo ^
  --copy-metadata protobuf ^
  --copy-metadata xformers ^
  --copy-metadata bitsandbytes ^
  --collect-submodules model ^
  --collect-submodules training ^
  --collect-submodules evolution ^
  --collect-submodules datasets ^
  --collect-submodules google.protobuf ^
  --collect-submodules unsloth ^
  --collect-submodules tiktoken ^
  --collect-submodules tiktoken_ext ^
  --collect-all unsloth ^
  --hidden-import torch ^
  --hidden-import torch.cuda ^
  --hidden-import filecmp ^
  --hidden-import tokenizers ^
  --hidden-import unsloth ^
  --hidden-import unsloth_zoo ^
  --hidden-import tiktoken_ext.openai_public ^
  train_sigma.py
if errorlevel 1 goto :fail

echo [build] Syncing runtime payload to project root...
if not exist "dist\eco_train\eco_train.exe" (
  echo [build][error] expected dist payload missing: dist\eco_train\eco_train.exe
  goto :fail
)
copy /y "dist\eco_train\eco_train.exe" "eco_train.exe" >nul
if errorlevel 1 goto :fail
if exist "dist\eco_train\_internal" (
  robocopy "dist\eco_train\_internal" "_internal" /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS >nul
  if errorlevel 8 goto :fail
)

echo.
echo [build] SUCCESS
echo [build] Exe path: %cd%\eco_train.exe
if exist "dist\eco_train" (
  rmdir /s /q "dist\eco_train"
)
if exist "eco_train.spec" (
  del /f /q "eco_train.spec"
)
if exist "build" (
  rmdir /s /q "build"
)
exit /b 0

:fail
echo.
echo [build] FAILED (errorlevel=%errorlevel%)
exit /b %errorlevel%
