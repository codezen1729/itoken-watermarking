# Local research runtime

The research adapter scores an actual, locally loaded trained model. It does not
download weights or call a paid API. The initial checkpoint was
[`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit/tree/8b403126fc14f14cfc99bb4cfa72ecbc129ea677),
revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`. This checkpoint uses 4-bit
quantization with group size 64. Its primary end-of-step token is the tokenizer's
EOS, `<|im_end|>` (151645). Keep the checkpoint outside the repository.

After the 1.5B runtime attempts below, the follow-up checkpoint is
[`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3),
revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Its packed weights are
approximately 278 MB. This is a smaller trained model with different prediction
probabilities and capabilities; it is not an execution-only change to the 1.5B
model. It was selected before development or held-out examples were evaluated.
Its `dense-fp32` run completed a six-token JSON smoke in 1.53 seconds and six
candidate-scoring contexts. These results are in
[qwen_05b_dense_fp32_v1](research_results/qwen_05b_dense_fp32_v1).
Recovery evaluation and watermark effectiveness require separate results.

After installing the research dependencies, the exact smaller checkpoint can be
downloaded explicitly with Hugging Face's library. This example stores it beside
the checkout; the experiment adapter itself performs no download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    revision="a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    local_dir="../itoken-models/qwen2.5-0.5b-instruct-4bit",
    allow_patterns=["*.safetensors", "*.json", "merges.txt"],
)
```

Use that local directory in the [research commands](README.md#research-extension-with-a-trained-model).

The Apple Silicon environment used for this work denies access to Metal. The
standard MLX wheel consequently failed during import, before Python could select
the CPU. We used MLX's supported CPU-only source configuration; no permissions
were relaxed. The upstream discussion describes
[this import limitation and CPU-only builds](https://github.com/ml-explore/mlx/issues/3148).

The CPU wheel was built and installed; model validation is recorded in the
experiment artifacts. The complete sanitized build provenance is in
[runtime_build.json](research_results/qwen_cpu_v1/runtime_build.json).
Build success alone is not evidence of a successful model experiment.

The quantized checkpoint passed a short CPU generation smoke test (13 tokens in
14.9 seconds). Its first scoring context was subsequently interrupted after
several minutes without a completed context. No development or test examples
were evaluated in that interrupted run.

A benchmark of the checkpoint's actual first 1536-by-1536 attention projection
identified the expensive operation. These are median kernel times, not full-model
throughput; the one-token timing used ten repeats and the 128-token timing two.
The [raw benchmark](research_results/qwen_cpu_v1/cpu_linear_benchmark.json)
records every timing and output difference.

| Execution | One token | 128 tokens |
| --- | ---: | ---: |
| Packed 4-bit weights | 0.405 ms | 49.0 ms |
| Dequantized BF16 | 0.263 ms | 0.539 ms |
| Dequantized FP32 | 0.0613 ms | 0.827 ms |

The full-model `--execution-mode dense-fp32` smoke was also interrupted before
generation completed; its status is retained in `research_results/qwen_dense_fp32_v1`.
The kernel speedup did not establish a practical full-model speedup.
The 1.5B dense-BF16 smoke reached decoding but was also interrupted because it
remained too slow. Neither dense attempt completed development or test cases.
Dense execution is a distinct variant.
It casts the checkpoint's affine quantization coefficients to FP32, expands the
packed weights, replaces quantized linear/embedding modules, and casts remaining
floating parameters to FP32. It does not recover the pre-quantization weights.
Its probabilities need not match packed quantized execution. The adapter records
the transformation and a SHA256 of the actual runtime parameters. FP32 parameter
storage alone was 6,174,857,216 bytes. `dense-bf16` applies the analogous
transformation in BF16 and approximately halves that storage. Measure full-model
speed and memory before expanding the pilot. The default adapter mode remains
`quantized`.

Prompt prefilling follows MLX-LM's `generate_step`: process all but the final
prompt token in chunks of 128, evaluate the cache states, then compute logits for
the final prompt token. This avoids computing unused vocabulary-head logits for
every prompt position. The chunk size and strategy are recorded in model
metadata. The independent numerical reference still evaluates the entire
prefix-plus-continuation sequence without a cache.
Across the six completed 0.5B scoring contexts, fourteen such comparisons had a
maximum token-log-probability difference of `2.69413e-5` nats and a maximum
sequence-log-probability difference of `7.65752e-5` nats. The open-instruction
context checked all nine candidate cache forks; the remaining contexts checked
their first candidate. These checks support numerical consistency of this
execution variant, not semantic coverage of the candidate set.

## Pinned build inputs

The build uses Python 3.13.3, CMake 4.4.5, setuptools 84.2.1,
typing-extensions 4.16.0, Apple Clang 21.0.0 (`clang-2100.1.1.101`), system
`make`, and Apple's Accelerate
framework. Ninja is unnecessary. The source archives below were obtained from
their official GitHub repositories at the linked immutable commits.

| Source | Exact archive | Archive SHA256 |
| --- | --- | --- |
| MLX 0.32.4 | [36969c09](https://codeload.github.com/ml-explore/mlx/tar.gz/36969c094354086e15d2a327a9c50feb20b9cf11) | `391b1e41414cddebe7a4fb5c55bf193ea5edd70e26adcfb0e1d6a5e5e86ec34a` |
| JSON 3.11.3 | [9cca280a](https://codeload.github.com/nlohmann/json/tar.gz/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03) | `0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a` |
| fmt 12.1.0 | [407c905e](https://codeload.github.com/fmtlib/fmt/tar.gz/407c905e45ad75fc29bf0f9bb7c5c2fd3475976f) | `2bc1fe4a5b6d5d6a614239b4ca1d520e66e152a02d3d262684d26dfd6ab3438a` |
| nanobind 3.1.0 | [68480a9e](https://codeload.github.com/wjakob/nanobind/tar.gz/68480a9e6883bb1bf65e1925423321e2a89d07f9) | `177ca257aacbf58004809f8cd1c20ad0416d8ace0a884c16a3c03ea87e81876b` |
| nanobind's robin-map submodule | [4ec1bf19](https://codeload.github.com/Tessil/robin-map/tar.gz/4ec1bf19c6a96125ea22062f38c2cf5b958e448e) | `4968947bd2e908854481519ec422823af4161a286ff1e16943293b753e80cb3f` |

The verified build-tool wheels were
`cmake-4.4.5-py3-none-macosx_10_10_universal2.whl`
(SHA256 `e07a5716b4233dc9c69d6ca6993f4405eefc7687e718d9f6eebbb6cc1be304ef`)
and `setuptools-84.2.1-py3-none-any.whl`
(SHA256 `ca140b45d5325a9478f880e351484f62fef15f336462e449ca0c03c873d589c2`).
These hashes were checked against each exact version's official PyPI metadata.
The resulting local CPU wheel,
`mlx-0.32.4-cp313-cp313-macosx_26_0_universal2.whl`, is 5,984,988 bytes;
its SHA256 is
`1d90b10e4ea31fd49d154b2ac136ba93c28b6b083cff068986d8ad947dec72a9`.

## CPU build recipe

From a checkout, create a separate environment and a `runtime/sources` directory.
Download and verify the archives above, then extract their top-level directories
as `runtime/sources/{mlx,json,fmt,nanobind}`. Extract robin-map into
`runtime/sources/nanobind/ext/robin_map`. Use a checkout path without spaces:
upstream `setup.py` splits `CMAKE_ARGS` on spaces.

```bash
python3.13 -m venv .venv-research
.venv-research/bin/python -m pip install \
  'cmake==4.4.5' 'setuptools==84.2.1' 'typing-extensions==4.16.0'
export ITOKEN_RUNTIME="$PWD/runtime"
export PATH="$PWD/.venv-research/bin:$PATH"
export PYPI_RELEASE=1
export MLX_BUILD_FRONTEND_PACKAGE=0
export MLX_BUILD_BACKEND_PACKAGE=0
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CMAKE_ARGS="-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON \
-DMLX_BUILD_CUDA=OFF -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_SAFETENSORS=ON \
-DMLX_BUILD_PYTHON_STUBS=OFF -DMLX_USE_CCACHE=OFF \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
-DFETCHCONTENT_SOURCE_DIR_JSON=$ITOKEN_RUNTIME/sources/json \
-DFETCHCONTENT_SOURCE_DIR_FMT=$ITOKEN_RUNTIME/sources/fmt \
-DFETCHCONTENT_SOURCE_DIR_NANOBIND=$ITOKEN_RUNTIME/sources/nanobind"
.venv-research/bin/python -m pip wheel --no-index --no-deps \
  --no-build-isolation runtime/sources/mlx -w runtime/cpu-wheelhouse
```

The local dependency overrides keep CMake from downloading additional sources.
`CMAKE_POLICY_VERSION_MINIMUM=3.5` is the documented
[CMake 4 compatibility option](https://cmake.org/cmake/help/latest/variable/CMAKE_POLICY_VERSION_MINIMUM.html)
for dependencies that declare older policies. The frontend/backend package flags
produce one complete CPU MLX wheel, rather than the normal split Metal package.

If this dedicated environment already contains standard MLX packages, uninstall
both `mlx` and `mlx-metal` from **that environment** before installing the built
wheel; the packages otherwise share backend paths. Install the local wheel with
`--no-deps`, then install the model tools:

```bash
.venv-research/bin/python -m pip install --no-deps runtime/cpu-wheelhouse/mlx-*.whl
.venv-research/bin/python -m pip install \
  'mlx-lm==0.32.2' 'transformers==5.24.0' 'numpy==2.5.5' \
  'tokenizers==0.23.3' 'huggingface-hub==2.4.0' \
  'hf-xet==1.8.0' 'sentencepiece==0.2.3'
.venv-research/bin/python -c \
  'import mlx.core as mx; print(mx.default_device(), mx.metal.is_available()); print(mx.sum(mx.array([1., 2.])).item())'
```

Record the resulting wheel hash and complete installed package list for each run.
Successful CPU import should precede a short model generation and candidate
scoring smoke test. Measure throughput before scheduling a large language
recovery evaluation. The adapter's `--device cpu` selects the CPU after import;
that option alone cannot repair a Metal wheel that fails during import.

Every scored result records prompt token IDs, terminated candidate token IDs,
individual learned token log probabilities, software versions, and checkpoint
hashes. Cached/uncached checks report numerical differences. The reported
next-iToken distribution remains conditional on the listed token sequences;
neither CPU execution nor a successful numerical check establishes coverage of
unrestricted language or robustness of an idea watermark.
