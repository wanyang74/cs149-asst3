#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// One CUDA thread processes one logical scan block.  At each upsweep
// level, the rightmost entry of the block becomes the sum of both halves.
__global__ void upsweep_kernel(int* output, int length, int half) {
    int logical_scan_block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int logical_scan_block_size = 2 * half;
    int logical_scan_block_start = logical_scan_block_id * logical_scan_block_size;

    if (logical_scan_block_start + logical_scan_block_size <= length) {
        int left_root = logical_scan_block_start + half - 1;
        int block_root = logical_scan_block_start + logical_scan_block_size - 1;
        output[block_root] += output[left_root];
    }
}

// During downsweep, block_root holds the prefix before the logical block and
// left_root still holds the total of its left half.  Distribute that prefix to
// the two children of the block.
__global__ void downsweep_kernel(int* output, int length, int half) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = 2 * half;
    int block_start = block_id * block_size;

    if (block_start + block_size <= length) {
        int left_root = block_start + half - 1;
        int block_root = block_start + block_size - 1;

        int left_sum = output[left_root];
        output[left_root] = output[block_root];
        output[block_root] += left_sum;
    }
}

// Mark every position i whose value equals the value immediately to its
// right.  flags[i] is therefore 1 when i should appear in the answer.
__global__ void mark_repeats_kernel(const int* input, int length, int* flags) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < length - 1) {
        flags[index] = (input[index] == input[index + 1]) ? 1 : 0;
    }
}

// flags[i] says whether i is a repeat; offsets[i] is the number of repeats
// before i.  Thus every matching index writes to a distinct compact slot.
__global__ void scatter_repeats_kernel(const int* flags, const int* offsets,
                                       int length, int* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < length - 1 && flags[index]) {
        output[offsets[index]] = index;
    }
}


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    if (N <= 0) {
        return;
    }

    const int length = nextPow2(N);

    // The caller may provide distinct input and output buffers.  The timing
    // harness happens to pre-initialize result, but copying here makes this
    // function correct for both in-place and out-of-place calls.
    if (input != result) {
        cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // The algorithm operates on a power-of-two-sized tree.  Padding with
    // zeros preserves every prefix sum in the original logical range.
    if (length > N) {
        cudaMemset(result + N, 0, (length - N) * sizeof(int));
    }

    // Upsweep: compute a sum at the root of every logical block.
    for (int half = 1; half < length; half *= 2) {
        int operations = length / (2 * half);
        int blocks = (operations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, length, half);
    }

    // The root represents the prefix before the whole array, which is zero.
    cudaMemset(result + length - 1, 0, sizeof(int));

    // Downsweep: replace subtree sums with their exclusive-prefix offsets.
    for (int half = length / 2; half >= 1; half /= 2) {
        int operations = length / (2 * half);
        int blocks = (operations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, length, half);
    }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    if (length <= 1) {
        return 0;
    }

    const int padded_length = nextPow2(length);
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (length - 1 + threads - 1) / threads;

    int* device_flags;
    int* device_offsets;
    cudaMalloc((void**)&device_flags, padded_length * sizeof(int));
    cudaMalloc((void**)&device_offsets, padded_length * sizeof(int));

    // Padding and the final element must contribute no repeat.  This also
    // prevents a padded value from being mistaken for a neighbor of the last
    // logical input element.
    cudaMemset(device_flags, 0, padded_length * sizeof(int));
    mark_repeats_kernel<<<blocks, threads>>>(device_input, length, device_flags);

    // Convert flags such as [0, 1, 0, 1] into write positions [0, 0, 1, 1].
    // Scan the padded range because exclusive_scan operates on a power of 2.
    exclusive_scan(device_flags, padded_length, device_offsets);
    scatter_repeats_kernel<<<blocks, threads>>>(device_flags, device_offsets,
                                                 length, device_output);

    // flags[length - 1] is zero, so its exclusive-scan value equals the total
    // number of valid repeat pairs.  Copying this one integer is necessary:
    // the C++ caller needs the count as a host-side return value.
    int repeat_count;
    cudaMemcpy(&repeat_count, device_offsets + length - 1, sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_flags);
    cudaFree(device_offsets);
    return repeat_count;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
