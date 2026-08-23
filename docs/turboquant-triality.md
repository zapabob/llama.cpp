# TurboQuant Triality runtime contract

This fork exposes a production runtime contract for TurboQuant Triality through
`<llama-turboquant.h>`. The supported multi-view graph is intentionally limited
to `LLM_ARCH_LLAMA`. Unsupported architectures, incomplete metadata, invalid
tensor layouts, and unavailable execution backends fail closed instead of
silently selecting another representation.

## Public API lifecycle

Applications first call `llama_tq_model_get_capabilities()` and inspect
`supported_execution_mask`. They then build a `llama_tq_context_config` with
public schema version 2 and create the context with
`llama_tq_init_from_model()`. The initializer validates and deep-copies the
configuration before reserving KV storage. `llama_tq_context_configure()` may
only make a storage-compatible change before the first successful encode or
decode. Configuration does not depend on process-global environment variables.

The four execution modes have distinct storage and graph behavior.

| Execution mode | K storage and attention behavior |
| --- | --- |
| `LLAMA_TQ_EXEC_SINGLE_VIEW` | One selected view per layer and one K cache. |
| `LLAMA_TQ_EXEC_BEST_PER_LAYER` | The active branch with the smallest finite `expected_error` is selected independently for each layer. |
| `LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS` | Three physical K caches feed one calibrated pre-softmax consensus, followed by one softmax and one V multiplication. Flash Attention is disabled and V must be non-quantized. |
| `LLAMA_TQ_EXEC_RESIDUAL_PARITY` | One packed K cache uses a four-bit parity-coupled physical code to represent the logical `[3, 1, 1]` sectors. The current production implementation is CPU-only and rejects KQV offload. |

For attention-logit consensus, branch `i` produces an unscaled raw dot product
`raw_i`. Calibration computes

```text
sum_i weight_i * ((raw_i - bias_i) / max(scale_i, 1e-6))
```

The existing attention softmax applies `1/sqrt(head_dim)` exactly once through
its normal KQ scale. `temperature` remains a required positive finite field for
ABI and metadata compatibility, but it is not applied by the consensus
operation.

Three-view KV state records an explicit layout marker and view count, persists
all three K tensors, and rejects restore into a one-view context. Sequence copy
operations cover every stored view. K shifts recover each view with its inverse
rotation, apply RoPE in the original K coordinates, and apply the forward
rotation before storing the result again. Single-view and best-per-layer shifts
use the rotation of the branch actually selected for that layer.

## GGUF schema namespaces

The artifact namespace and public runtime namespace are independent.
`tq_schema_version` is a `uint32` artifact version and must currently equal 1.
`hypura.turboquant.schema_version` is a `uint32` public contract selector.
An absent public key, or public value 1, selects the legacy parser. Public value
2 selects Triality schema v2. Artifact value 2 is unsupported and never selects
the public v2 parser. Wrong GGUF types and unsupported values are errors.

Schema v2 uses the canonical view order `vector`, `spinor_plus_proxy`, and
`spinor_minus_proxy`. Per-layer rotations are F32 square matrices stored below
`turboquant.profile.<profile_id>.layer.<layer>.rotation.<view>`. Consensus
weights, bias, scale, and temperature are F32 tensors below
`turboquant.profile.<profile_id>.consensus.*`. The two spinor names describe
finite-dimensional proxy views; they do not claim exact spinor or universal
representation equivalence.

Residual parity requires profile and mode
`key_only_block_so8_triality_residual_parity`. Its public metadata lives below
`hypura.turboquant.triality.residual_parity` and supplies `sector_bits`,
`fixed_sector_scales`, `beta`, `payload_bits_per_channel_milli`, and
`controller_bytes`. The logical schema remains `[3, 1, 1]` and reports 5.000
logical bits per channel. It does not physically store two independent residual
bits. Each channel instead occupies four physical bits: bits `[2:0]` hold a
signed main code from -3 through 3, with code 4 reserved, and bit 3 holds one
residual selector. Decode expands the logical residual bits as
`plus = selector` and `minus = selector XOR (main_code & 1)`.

Each layer has a separate 51-byte controller tensor. A packed row occupies
`ceil(logical_channels * 4 / 8)` bytes. Production KV-cache construction checks
the controller-inclusive physical budget per layer:

```text
((rows * row_bytes + 51) * 8) / (rows * logical_channels) <= 5
```

Shapes too small to amortize the controller while satisfying this inequality
fail closed during cache construction. The logical 5-bit sector contract and
the controller-inclusive physical total are therefore reported separately.
The final real-GGUF context check exercised one layer with 512 physical rows
and 32 logical channels. Its single physical K view used an 8,192-byte packed
payload plus the 51-byte controller, or 8,243 bytes total and
4.02490234375 controller-inclusive bits per channel. All 34 residual-context
assertions passed.

Unused high bits in the final packed byte and reserved controller bytes are
zero. State restore rejects non-zero padding, the reserved main-sector code,
controller mismatches, and incompatible execution layouts. K shifts decode the
stored representation, apply RoPE in the original K coordinates, and encode it
again; re-encoding restores deterministic zero padding.

## NC-KA, URT, and telemetry limits

Schema v2 validates the NC-KA and URT metadata namespaces, but that validation
must not be read as a production execution claim. The current llama runtime
reports `ncka_available=false` and `urt_available=false`. An optional,
unsupported NC-KA controller may select its declared static fallback weights;
`ncka_static_fallback_selected` reports that metadata decision. A required,
unsupported NC-KA controller fails closed. The standalone consensus operator
does not execute the NC-KA controller, so its `ka_fallback_used` runtime metric
is unavailable rather than evidence that the controller ran.

URT manifest hashes and representation declarations are validated at load time,
but the standalone operator has no production URT execution path. Consequently,
operator-word hashes and URT runtime telemetry are unavailable. The public
telemetry getter also returns unavailable when tracing is disabled or when no
sample has been recorded; the standalone operator evidence below is test JSON,
not a recorded `llama_tq_context_get_last_metrics()` sample.

## CUDA target and numerical contract

An RTX 50-series build uses the standard CUDA target without architecture-
accelerated suffixes:

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
```

Plain standard `sm_120` uses generic fallback code for block-scale operations
whose native accelerated forms are not available under that target. The stable
artifact makes no claim beyond standard `sm_120`. Its build proof records
`arch=compute_120,code=[compute_120,sm_120]`; `cuobjdump` inspection of the
release CUDA DLL reports `sm_120` as the only embedded architecture. Triality
consensus has a dedicated CUDA kernel for this target. Residual parity remains
CPU-only and is rejected when its required placement cannot be honored. CPU and
CUDA consensus tests use a relative tolerance of `1e-4` across dense F32/F16 K,
Turbo2/3/4 K, dense or packed rotations, GQA, and stream broadcasting.

The release evidence was captured on an NVIDIA GeForce RTX 5060 Ti with compute
capability 12.0 and CUDA compiler 12.8.61. The test is a synchronized standalone
GGML attention-score operator benchmark, not an end-to-end llama throughput
measurement. It uses five warmup iterations and twenty measured repetitions.

| Evidence | Shape | Mean | p50 | p95 |
| --- | --- | ---: | ---: | ---: |
| Prefill latency | `q=16,kv=64,h=4,s=1,d=128`, F32 | 0.338675 ms | 0.3058 ms | 0.4684 ms |
| Decode latency | `q=1,kv=64,h=4,s=1,d=128`, F32 | 0.05053 ms | 0.0358 ms | 0.0526 ms |

The maximum CPU/CUDA attention-output Frobenius relative error was
`1.33893899e-07`. The memory snapshot reported 17,102,733,312 total VRAM bytes
and 15,907,946,496 free VRAM bytes. Per-operation peak VRAM is `null` because
the backend API exposes no per-operation peak counter. Next-logit KL divergence
and hidden-state cosine similarity are also `null`: this standalone operator
produces neither model logits nor hidden states.

The corresponding release proof files are
`tq-triality-consensus-sm120-build-proof.txt`,
`tq-triality-consensus-sm120-gpu.jsonl`, and
`tq-triality-consensus-sm120-sha256.txt`.

Identity-view fallback is development-only and must be explicitly enabled.
Production schema-v2 multi-view models are expected to provide all rotations.
