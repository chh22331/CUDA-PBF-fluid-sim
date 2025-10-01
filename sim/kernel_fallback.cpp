#include <cstdint>
#include <cstdio>

extern "C" int run_vector_add_cpu(const int* a, const int* b, int* c, unsigned int size)
{
    for (unsigned i = 0; i < size; ++i) c[i] = a[i] + b[i];
    // mimic cudaSuccess (0)
    return 0;
}
