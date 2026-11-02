// scan_sim.cpp -- CPU simulator for testing scan/find_repeats indexing on a Mac (no GPU).
//
// Kernels are written as plain functions taking (blockIdx, blockDim, threadIdx), so a
// kernel body can be copied verbatim into scan.cu. `launch` runs every (block, thread)
// pair in a *shuffled* order: if your result depends on thread order within one launch,
// the test will (likely) catch it, just like a real race would bite on the GPU.
//
// Build & run:  clang++ -std=c++17 -O2 -o scan_sim scan_sim.cpp && ./scan_sim

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

struct dim { int x; };
#define KERNEL_ARGS dim blockIdx, dim blockDim, dim threadIdx

static std::mt19937 rng(149);

template <typename F>
void launch(int blocks, int threadsPerBlock, F kernel) {
    std::vector<std::pair<int, int>> order;
    for (int b = 0; b < blocks; b++)
        for (int t = 0; t < threadsPerBlock; t++)
            order.push_back({b, t});
    std::shuffle(order.begin(), order.end(), rng);
    for (auto [b, t] : order)
        kernel(dim{b}, dim{threadsPerBlock}, dim{t});
}

#define THREADS_PER_BLOCK 256

static int nextPow2(int n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// ---------------------------------------------------------------------------
// CS149 TODO: kernels. Write the bodies exactly as you would in CUDA.
// Add/rename parameters as you like; mirror any change in the host code below.
// ---------------------------------------------------------------------------

void upsweep_kernel(KERNEL_ARGS, int N, int two_d, int* data) {
}

void downsweep_kernel(KERNEL_ARGS, int N, int two_d, int* data) {
}

// ---------------------------------------------------------------------------
// CS149 TODO: host code, mirroring exclusive_scan() in scan.cu.
// `result` already holds a copy of the input and has nextPow2(N) slots; slots >= N
// contain garbage (as on the GPU). In scan.cu, replace `launch(blocks, tpb, [&](...){...})`
// with `kernel<<<blocks, tpb>>>(...)`.
// ---------------------------------------------------------------------------

void exclusive_scan(int* input, int N, int* result) {
    (void)input;
    // Example launch:
    // launch(blocks, THREADS_PER_BLOCK, [&](KERNEL_ARGS) { upsweep_kernel(blockIdx, blockDim, threadIdx, N, two_d, result); });
}

// ---------------------------------------------------------------------------
// CS149 TODO: find_repeats. `device_input` and `device_output` have nextPow2(length)
// slots. Allocate temporaries with std::vector here (cudaMalloc on the GPU).
// Return the number of repeats; write their indices to device_output.
// ---------------------------------------------------------------------------

int find_repeats(int* device_input, int length, int* device_output) {
    (void)device_input; (void)length; (void)device_output;
    return 0;
}

// ---------------------------------------------------------------------------
// Tests (no need to edit)
// ---------------------------------------------------------------------------

static const int GARBAGE = 0x5eed5eed;

static bool test_scan(int N) {
    int P = nextPow2(N);
    std::vector<int> in(P, GARBAGE), out(P, GARBAGE);
    for (int i = 0; i < N; i++) in[i] = out[i] = rng() % 1000 - 500;
    exclusive_scan(in.data(), N, out.data());
    int sum = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != sum) {
            printf("  scan N=%d: mismatch at %d: got %d, want %d\n", N, i, out[i], sum);
            return false;
        }
        sum += in[i];
    }
    return true;
}

static bool test_find_repeats(int N, int range) {
    int P = nextPow2(N);
    std::vector<int> in(P, GARBAGE), out(P, GARBAGE);
    for (int i = 0; i < N; i++) in[i] = rng() % range;
    if (N < P) in[N] = in[N - 1];  // a repeat that straddles the logical end must NOT count
    std::vector<int> want;
    for (int i = 0; i + 1 < N; i++)
        if (in[i] == in[i + 1]) want.push_back(i);
    int got = find_repeats(in.data(), N, out.data());
    if (got != (int)want.size()) {
        printf("  find_repeats N=%d: count %d, want %zu\n", N, got, want.size());
        return false;
    }
    for (int i = 0; i < got; i++)
        if (out[i] != want[i]) {
            printf("  find_repeats N=%d: out[%d]=%d, want %d\n", N, i, out[i], want[i]);
            return false;
        }
    return true;
}

int main() {
    std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 8, 9, 31, 255, 256, 257, 511, 512, 513,
                              1000, 1023, 1024, 1025, 4096, 100000, 1 << 20, 1000003};
    int pass = 0, fail = 0;
    for (int n : sizes) (test_scan(n) ? pass : fail)++;
    printf("scan:         %d passed, %d failed\n", pass, fail);
    pass = fail = 0;
    for (int n : sizes) {
        (test_find_repeats(n, 3) ? pass : fail)++;    // many repeats
        (test_find_repeats(n, 1000) ? pass : fail)++; // few repeats
    }
    {   // README example: {1,2,2,1,1,1,3,5,3,3} -> {1,3,4,8}
        std::vector<int> in = {1, 2, 2, 1, 1, 1, 3, 5, 3, 3, 0, 0, 0, 0, 0, 0}, out(16, GARBAGE);
        int got = find_repeats(in.data(), 10, out.data());
        bool ok = got == 4 && out[0] == 1 && out[1] == 3 && out[2] == 4 && out[3] == 8;
        if (!ok) printf("  find_repeats README example failed (count %d)\n", got);
        (ok ? pass : fail)++;
    }
    printf("find_repeats: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
