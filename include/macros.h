#pragma once
#include <cstdio>
#define gpuChkerr(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf("Error (%d) %s at %s:%d\n", (int)x, cudaGetErrorString(x),  \
                   __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define randChkerr(x)                                                          \
    do {                                                                       \
        if ((x) != CURAND_STATUS_SUCCESS) {                                    \
            printf("Error (%d) at %s:%d\n", (int)x, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define blasChkerr(x)                                                          \
    do {                                                                       \
        if ((x) != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("Error (%d) at %s:%d\n", (int)x, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define KERNEL static __global__ void
