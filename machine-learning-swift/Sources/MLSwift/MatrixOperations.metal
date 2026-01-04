/// MatrixOperations.metal
/// Metal compute shaders for GPU-accelerated matrix operations
/// Optimized for Apple Silicon (M1/M2/M3+)

#include <metal_stdlib>
using namespace metal;

/// Element-wise matrix addition: C = A + B
kernel void matrix_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] + B[id];
}

/// Element-wise matrix subtraction: C = A - B
kernel void matrix_subtract(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] - B[id];
}

/// Element-wise matrix scaling: C = A * scale
kernel void matrix_scale(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] * scale;
}

/// Matrix multiplication: C = A * B
/// Optimized for Apple Silicon with tiling
kernel void matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],  // A rows
    constant uint& K [[buffer(4)]],  // A cols / B rows
    constant uint& N [[buffer(5)]],  // B cols
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

/// Matrix multiplication with transpose A: C = A^T * B
kernel void matrix_multiply_transpose_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],  // A rows (K before transpose)
    constant uint& M [[buffer(4)]],  // A cols (M after transpose)
    constant uint& N [[buffer(5)]],  // B cols
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[k * M + row] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

/// Matrix multiplication with transpose B: C = A * B^T
kernel void matrix_multiply_transpose_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],  // A rows
    constant uint& K [[buffer(4)]],  // A cols / B cols
    constant uint& N [[buffer(5)]],  // B rows (N after transpose)
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

/// ReLU activation: C = max(0, A)
kernel void relu_forward(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = max(0.0f, A[id]);
}

/// ReLU gradient: grad_input += (input > 0) ? grad_output : 0
kernel void relu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    grad_input[id] += (input[id] > 0.0f) ? grad_output[id] : 0.0f;
}

/// Softmax forward pass (two-stage: find max and compute exp/sum)
/// Stage 1: Find maximum value (reduction)
kernel void softmax_max_reduce(
    device const float* input [[buffer(0)]],
    device float* max_val [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float shared_max[256];
    
    // Load and find local max
    float local_max = -INFINITY;
    for (uint i = tid; i < size; i += 256) {
        local_max = max(local_max, input[i]);
    }
    shared_max[lid] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        max_val[bid] = shared_max[0];
    }
}

/// Stage 2: Compute exp and sum
kernel void softmax_exp_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* sum [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float shared_sum[256];
    
    // Compute exp and local sum
    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += 256) {
        float exp_val = exp(input[i] - max_val);
        output[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[lid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sum in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        sum[0] = shared_sum[0];
    }
}

/// Stage 3: Normalize by sum
kernel void softmax_normalize(
    device float* output [[buffer(0)]],
    constant float& sum [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] /= sum;
}

/// Cross-entropy loss: output = -p * log(q)
kernel void cross_entropy_forward(
    device const float* p [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    // Avoid log(0) by using a small epsilon
    float q_val = max(q[id], 1e-7f);
    output[id] = (p[id] == 0.0f) ? 0.0f : -p[id] * log(q_val);
}

/// Cross-entropy gradient w.r.t. q: grad_q += -p / q * grad
kernel void cross_entropy_backward_q(
    device const float* p [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* grad [[buffer(2)]],
    device float* grad_q [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float q_val = max(q[id], 1e-7f);
    grad_q[id] += -p[id] / q_val * grad[id];
}

/// Cross-entropy gradient w.r.t. p: grad_p += -log(q) * grad
kernel void cross_entropy_backward_p(
    device const float* q [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* grad_p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float q_val = max(q[id], 1e-7f);
    grad_p[id] += -log(q_val) * grad[id];
}
