#pragma once

#include "common.cuh"

void ggml_cuda_tq_triality_kq_consensus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
