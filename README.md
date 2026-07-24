# onnxruntime-gpu for NVIDIA GB10 / DGX Spark — aarch64 + CUDA 13 + sm_121

Prebuilt `onnxruntime-gpu` wheels for **aarch64 (SBSA) with the CUDA Execution Provider
compiled for compute capability 12.1 (`sm_121`)** — the NVIDIA GB10 in the DGX Spark.

| | |
|---|---|
| ONNX Runtime | 1.27.1 |
| Python | 3.12 (`cp312`) |
| Platform | `linux_aarch64` (SBSA) |
| CUDA | 13.x (built against 13.0.2, cuDNN 9) |
| GPU archs | `90a-real;120a-real;121-real` — GB10/Spark, plus Hopper and RTX 50-series |
| sha256 | `5829acda0a14e4e9a95161ec0d2bb56f515e029e6ef4da39b7bbc00cd6de8faf` |

## Install

```sh
pip install https://github.com/seitzbg/onnxruntime-gpu-sm121-aarch64/releases/download/v1.27.1/onnxruntime_gpu-1.27.1-cp312-cp312-linux_aarch64.whl
```

## Verify — and don't trust `get_available_providers()`

```python
import onnxruntime as ort
s = ort.InferenceSession("your_model.onnx",
                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(s.get_providers())        # must CONTAIN CUDAExecutionProvider
s.run(None, feed)               # must not raise
```

`ort.get_available_providers()` lists providers **compiled in**, not providers that
successfully *loaded*. A broken CUDA build still reports `CUDAExecutionProvider` there
while silently running every op on CPU. Always check `session.get_providers()` and
confirm a real speedup.

## Why this exists

There is no `onnxruntime-gpu` you can install for this hardware:

- **PyPI** ships `onnxruntime-gpu` only for `manylinux_2_28_x86_64` and `win_amd64` — no
  aarch64 build at any version. It is not on `pypi.nvidia.com` either (404).
- **conda-forge** *does* ship `onnxruntime-*-cuda130` for `linux-aarch64`/`sbsa`. It
  installs, and a `CUDAExecutionProvider` session is created — then the first `run()` dies:

  ```
  CUDA error cudaErrorNoKernelImageForDevice:
  no kernel image is available for execution on the device
  ```

  The feedstock builds `CUDA_ARCH_LIST="75-real;…;100-real;120"` — no `121`, and the
  `compute_120` PTX does not JIT onto `sm_121`.
- **CUDA 12.8 and earlier cannot target `sm_121` at all** — it needs CUDA 12.9+.

`cupy-cuda13x` and `cuml-cu13` both ship working aarch64 wheels, so ONNX Runtime is the
only gap in an otherwise complete CUDA 13 stack on this hardware.

## Build gotcha — an arch list of only 120/121 silently produces a broken .so

> **Fixed upstream, but not yet in any release.** ONNX Runtime commit
> [`ed98916`](https://github.com/microsoft/onnxruntime/commit/ed9891635) (PR
> [#29824](https://github.com/microsoft/onnxruntime/pull/29824), 2026-07-24) makes
> `EXCLUDE_SM120_REAL` apply only on MSVC, so Linux builds keep the SM120 archs. The
> problem below affects **v1.27.1 and earlier** — i.e. every released version at time of
> writing. It is why these wheels are built with `90;120;121`.

If you build this yourself, **your arch list must include an architecture in `[75, 120)`**.
Building for only Blackwell (`121`, or `120;121`) yields a wheel that compiles and installs
cleanly but whose CUDA provider cannot load:

```
Failed to load library libonnxruntime_providers_cuda.so with error: undefined symbol:
_ZNK11onnxruntime3llm7kernels15cutlass_kernels13MoeGemmRunnerIffffE10getConfigsEv
Failed to create CUDAExecutionProvider.
```

The chain, in ONNX Runtime 1.27.1:

1. `cmake/external/cuda_configuration.cmake:206-217` normalizes the arch list. `120` is in
   `ARCHITECTURES_WITH_ACCEL` so it becomes `120a-real`; `121` is not, so it becomes
   `121-real`.
2. `cmake/onnxruntime_providers_cuda.cmake:531` calls
   `onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75 EXCLUDE_SM120_REAL)`.
3. That filter (`cmake/onnxruntime_cuda_source_filters.cmake:178`) drops every arch that is
   `>= 120` **and** ends in `-real` — which is *both* entries.
4. `_ort_llm_cuda_architectures` is empty, so the `onnxruntime_providers_cuda_llm` object
   library is never created and `contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_*.cu` are
   never compiled…
5. …but `libonnxruntime_providers_cuda.so` still references their symbols. It links, and
   fails at `dlopen`.

Adding `90` keeps that object library alive. These wheels are built with `90;120;121`.

On current `main` this is resolved differently — the filter no longer excludes SM120 real
archs on non-MSVC toolchains, so `120;121` alone would work once that ships in a release.

## Build provenance

Built from unmodified [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
`v1.27.1` source, inside `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` on an aarch64 host:

```sh
./build.sh --config Release --use_cuda \
    --cuda_home /usr/local/cuda --cudnn_home /usr \
    --build_wheel --skip_tests --parallel 12 --nvcc_threads 2 \
    --allow_running_as_root --compile_no_warning_as_error \
    --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=90;120;121' \
                          onnxruntime_BUILD_UNIT_TESTS=OFF
```

Verified on real hardware (NVIDIA GB10, driver 595.71.05, CUDA 13.2) against a production
ONNX model — CUDA EP active, **7.3x** faster than the CPU EP on the same session.

## License

ONNX Runtime is MIT licensed by Microsoft. These are unmodified builds of upstream source;
the MIT license and copyright travel in the wheel. This repository distributes build
artifacts only and claims no additional rights over them.
