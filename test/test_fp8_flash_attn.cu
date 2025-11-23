#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include "flash.h"
#include "cutlass/numeric_types.h"

// For FP8 testing
using FP8Type = cutlass::float_e4m3_t;
using HalfType = flash_attn::half;

// Reference attention implementation for validation
void reference_attention(const HalfType* Q, const HalfType* K, const HalfType* V, HalfType* O,
                        int batch_size, int num_heads, int seqlen_q, int seqlen_k, int head_dim,
                        float scale, bool is_causal) {
    std::vector<float> scores(seqlen_q * seqlen_k);
    std::vector<float> attn_weights(seqlen_q * seqlen_k);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Compute Q @ K^T
            for (int i = 0; i < seqlen_q; i++) {
                for (int j = 0; j < seqlen_k; j++) {
                    if (is_causal && j > i) {
                        scores[i * seqlen_k + j] = -INFINITY;
                        continue;
                    }

                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int q_idx = b * num_heads * seqlen_q * head_dim + h * seqlen_q * head_dim + i * head_dim + d;
                        int k_idx = b * num_heads * seqlen_k * head_dim + h * seqlen_k * head_dim + j * head_dim + d;
                        score += static_cast<float>(Q[q_idx]) * static_cast<float>(K[k_idx]);
                    }
                    scores[i * seqlen_k + j] = score * scale;
                }
            }

            // Softmax
            for (int i = 0; i < seqlen_q; i++) {
                float max_score = -INFINITY;
                for (int j = 0; j < seqlen_k; j++) {
                    max_score = std::max(max_score, scores[i * seqlen_k + j]);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < seqlen_k; j++) {
                    attn_weights[i * seqlen_k + j] = expf(scores[i * seqlen_k + j] - max_score);
                    sum_exp += attn_weights[i * seqlen_k + j];
                }

                for (int j = 0; j < seqlen_k; j++) {
                    attn_weights[i * seqlen_k + j] /= sum_exp;
                }
            }

            // Compute attention @ V
            for (int i = 0; i < seqlen_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float output = 0.0f;
                    for (int j = 0; j < seqlen_k; j++) {
                        int v_idx = b * num_heads * seqlen_k * head_dim + h * seqlen_k * head_dim + j * head_dim + d;
                        output += attn_weights[i * seqlen_k + j] * static_cast<float>(V[v_idx]);
                    }
                    int o_idx = b * num_heads * seqlen_q * head_dim + h * seqlen_q * head_dim + i * head_dim + d;
                    O[o_idx] = static_cast<HalfType>(output);
                }
            }
        }
    }
}

// Compute relative error (comparing FP8 output to FP16 reference)
float compute_relative_error(const FP8Type* a, const HalfType* b, int size) {
    float max_diff = 0.0f;
    float max_val = 0.0f;
    int nan_count = 0;

    for (int i = 0; i < size; i++) {
        float val_a = static_cast<float>(a[i]);
        float val_b = static_cast<float>(b[i]);

        if (std::isnan(val_a) || std::isinf(val_a)) {
            nan_count++;
            continue;
        }

        float diff = std::abs(val_a - val_b);
        max_diff = std::max(max_diff, diff);
        max_val = std::max(max_val, std::max(std::abs(val_a), std::abs(val_b)));
    }

    if (nan_count > 0) {
        printf("WARNING: Found %d NaN/Inf values in output!\n", nan_count);
        return 1.0f;  // Return error = 1.0 for NaN
    }

    return max_diff / (max_val + 1e-6f);
}

void test_flash_attention() {
    printf("=== Testing FP8 E4M3 Flash Attention ===\n\n");

    // Test configuration
    const int batch_size = 2;
    const int num_heads = 8;
    const int seqlen_q = 128;
    const int seqlen_k = 128;
    const int head_dim = 64;
    const bool is_causal = true;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Num heads: %d\n", num_heads);
    printf("  Sequence length (Q): %d\n", seqlen_q);
    printf("  Sequence length (K): %d\n", seqlen_k);
    printf("  Head dimension: %d\n", head_dim);
    printf("  Causal: %s\n", is_causal ? "true" : "false");
    printf("  Scale: %f\n\n", scale);

    // Allocate host memory
    const int q_size = batch_size * num_heads * seqlen_q * head_dim;
    const int k_size = batch_size * num_heads * seqlen_k * head_dim;
    const int v_size = batch_size * num_heads * seqlen_k * head_dim;
    const int o_size = batch_size * num_heads * seqlen_q * head_dim;

    std::vector<FP8Type> h_q(q_size);
    std::vector<FP8Type> h_k(k_size);
    std::vector<FP8Type> h_v(v_size);
    std::vector<FP8Type> h_o(o_size);
    std::vector<HalfType> h_o_ref(o_size);

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);

    printf("Initializing random input data...\n");
    for (int i = 0; i < q_size; i++) h_q[i] = FP8Type(dist(gen) * 0.1f);
    for (int i = 0; i < k_size; i++) h_k[i] = FP8Type(dist(gen) * 0.1f);
    for (int i = 0; i < v_size; i++) h_v[i] = FP8Type(dist(gen) * 0.1f);

    // Allocate device memory
    FP8Type *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size * sizeof(FP8Type));
    cudaMalloc(&d_k, k_size * sizeof(FP8Type));
    cudaMalloc(&d_v, v_size * sizeof(FP8Type));
    cudaMalloc(&d_o, o_size * sizeof(FP8Type));

    // Initialize output to test pattern
    cudaMemset(d_o, 0xFF, o_size * sizeof(FP8Type));

    // Copy data to device
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(FP8Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(FP8Type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(FP8Type), cudaMemcpyHostToDevice);

    printf("Input data copied to device\n");
    printf("Q first 5 values: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", static_cast<float>(h_q[i]));
    printf("\n");

    // Setup strides (contiguous layout)
    int q_batch_stride = num_heads * seqlen_q * head_dim;
    int q_head_stride = seqlen_q * head_dim;
    int q_row_stride = head_dim;
    int k_batch_stride = num_heads * seqlen_k * head_dim;
    int k_head_stride = seqlen_k * head_dim;
    int k_row_stride = head_dim;
    int v_batch_stride = num_heads * seqlen_k * head_dim;
    int v_head_stride = seqlen_k * head_dim;
    int v_row_stride = head_dim;
    int o_batch_stride = num_heads * seqlen_q * head_dim;
    int o_head_stride = seqlen_q * head_dim;
    int o_row_stride = head_dim;

    printf("Running Flash Attention...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate and initialize descale factors (set to 1.0 for no scaling)
    float *d_q_descale, *d_k_descale, *d_v_descale;
    cudaMalloc(&d_q_descale, sizeof(float));
    cudaMalloc(&d_k_descale, sizeof(float));
    cudaMalloc(&d_v_descale, sizeof(float));

    float descale_val = 1.0f;
    cudaMemcpy(d_q_descale, &descale_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_descale, &descale_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_descale, &descale_val, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = 0;
    cudaEventRecord(start, stream);

    // Call FP8 Flash Attention
    {
        printf("Calling flash_attention_fp8_forward...\n");
        flash_attn::flash_attention_fp8_forward(
            d_q, d_k, d_v, d_o,
            d_q_descale, d_k_descale, d_v_descale,
            batch_size, seqlen_q, seqlen_k, num_heads, num_heads, head_dim,
            q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride,
            q_head_stride, k_head_stride, v_head_stride, o_head_stride,
            q_row_stride, k_row_stride, v_row_stride, o_row_stride,
            scale, is_causal, -1, -1, stream
        );

        // Check for launch errors
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("CUDA Launch Error: %s\n", cudaGetErrorString(launch_err));
            goto cleanup;
        }

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        // Check for execution errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Execution Error: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        printf("Flash Attention kernel completed successfully\n");
    }

    {
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        printf("Flash Attention completed in %.3f ms\n", elapsed_ms);

        // Copy output back
        cudaMemcpy(h_o.data(), d_o, o_size * sizeof(FP8Type), cudaMemcpyDeviceToHost);

        // Compute reference using FP16 for better accuracy
        printf("\nComputing reference solution...\n");
        // Convert FP8 to FP16 for reference
        std::vector<HalfType> h_q_fp16(q_size);
        std::vector<HalfType> h_k_fp16(k_size);
        std::vector<HalfType> h_v_fp16(v_size);
        for (int i = 0; i < q_size; i++) h_q_fp16[i] = static_cast<HalfType>(static_cast<float>(h_q[i]));
        for (int i = 0; i < k_size; i++) h_k_fp16[i] = static_cast<HalfType>(static_cast<float>(h_k[i]));
        for (int i = 0; i < v_size; i++) h_v_fp16[i] = static_cast<HalfType>(static_cast<float>(h_v[i]));

        reference_attention(h_q_fp16.data(), h_k_fp16.data(), h_v_fp16.data(), h_o_ref.data(),
                           batch_size, num_heads, seqlen_q, seqlen_k, head_dim,
                           scale, is_causal);

        // Compare results
        float rel_error = compute_relative_error(h_o.data(), h_o_ref.data(), o_size);
        printf("Relative error: %.6f\n", rel_error);

        // Print sample outputs
        printf("\nSample outputs (first 10 values):\n");
        printf("Flash Attention: ");
        for (int i = 0; i < 10; i++) {
            printf("%.4f ", static_cast<float>(h_o[i]));
        }
        printf("\n");

        printf("Reference:       ");
        for (int i = 0; i < 10; i++) {
            printf("%.4f ", static_cast<float>(h_o_ref[i]));
        }
        printf("\n");

        // Validation - FP8 has lower precision, so use higher tolerance
        const float tolerance = 0.05f;  // 5% tolerance for FP8
        if (rel_error < tolerance) {
            printf("\n✓ TEST PASSED (relative error < %.3f)\n", tolerance);
        } else {
            printf("\n✗ TEST FAILED (relative error %.6f >= %.3f)\n", rel_error, tolerance);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

cleanup:
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_q_descale);
    cudaFree(d_k_descale);
    cudaFree(d_v_descale);
}

int main() {
    printf("Flash Attention Test Suite\n");
    printf("===========================\n\n");

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }

    printf("Found %d CUDA device(s)\n\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  SMs: %d\n\n", prop.multiProcessorCount);
    }

    // Run test
    test_flash_attention();

    return 0;
}
