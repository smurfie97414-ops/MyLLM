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
set "REQ_FILE=requirements-py313-cu128.txt"
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
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 goto :fail
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

for /f "usebackq delims=" %%I in (`%PYTHON% -c "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['include']))"`) do set "PY_INCLUDE_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import pathlib,sys; print(pathlib.Path(sys.base_exec_prefix) / 'libs')"`) do set "PY_LIB_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import torch,pathlib; print(pathlib.Path(torch.__file__).resolve().parent)"`) do set "TORCH_PKG_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import wandb,pathlib; print(pathlib.Path(wandb.__file__).resolve().parent / 'vendor' / 'gql-0.2.0')"`) do set "WANDB_GQL_VENDOR_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import wandb,pathlib; print(pathlib.Path(wandb.__file__).resolve().parent / 'vendor' / 'graphql-core-1.1')"`) do set "WANDB_GRAPHQL_VENDOR_DIR=%%I"
for /f "usebackq delims=" %%I in (`%PYTHON% -c "import wandb,pathlib; print(pathlib.Path(wandb.__file__).resolve().parent / 'vendor' / 'promise-2.3.0')"`) do set "WANDB_PROMISE_VENDOR_DIR=%%I"
set "TORCH_INCLUDE_DIR=%TORCH_PKG_DIR%\include"
set "TORCH_LIB_DIR=%TORCH_PKG_DIR%\lib"
echo [build] Python include dir: %PY_INCLUDE_DIR%
echo [build] Python libs dir: %PY_LIB_DIR%
echo [build] Torch package dir: %TORCH_PKG_DIR%
echo [build] Torch include dir: %TORCH_INCLUDE_DIR%
echo [build] Torch lib dir: %TORCH_LIB_DIR%
echo [build] wandb gql vendor dir: %WANDB_GQL_VENDOR_DIR%
echo [build] wandb graphql vendor dir: %WANDB_GRAPHQL_VENDOR_DIR%
echo [build] wandb promise vendor dir: %WANDB_PROMISE_VENDOR_DIR%

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
  --paths . ^
  --paths "%WANDB_GQL_VENDOR_DIR%" ^
  --paths "%WANDB_GRAPHQL_VENDOR_DIR%" ^
  --paths "%WANDB_PROMISE_VENDOR_DIR%" ^
  --add-data "model\triton_kernel_src.py;model" ^
  --add-data "model\sigma_kernel_src.py;model" ^
  --add-data "training\optimizer_gn_kernel_src.py;training" ^
  --add-data "training\sigma_trainer.py;training" ^
  --add-data "evolution\meta_loop.py;evolution" ^
  --add-data "%PY_INCLUDE_DIR%;Include" ^
  --add-data "%PY_LIB_DIR%;libs" ^
  --add-data "%TORCH_INCLUDE_DIR%;torch\include" ^
  --add-data "%TORCH_LIB_DIR%;torch\lib" ^
  --collect-data tiktoken ^
  --collect-data trl ^
  --collect-data unsloth ^
  --collect-data triton ^
  --collect-submodules model ^
  --collect-submodules training ^
  --collect-submodules evolution ^
  --collect-submodules datasets ^
  --collect-submodules transformers ^
  --collect-submodules trl ^
  --collect-submodules wandb_gql ^
  --collect-submodules wandb_graphql ^
  --collect-submodules wandb_promise ^
  --collect-submodules unsloth ^
  --collect-submodules unsloth_zoo ^
  --collect-submodules triton ^
  --collect-submodules tiktoken ^
  --collect-submodules tiktoken_ext ^
  --hidden-import torch ^
  --hidden-import torch.cuda ^
  --hidden-import trl ^
  --hidden-import wandb ^
  --hidden-import wandb_gql ^
  --hidden-import wandb_graphql ^
  --hidden-import wandb_promise ^
  --hidden-import unsloth ^
  --hidden-import unsloth_zoo ^
  --hidden-import triton ^
  --hidden-import triton.language ^
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
