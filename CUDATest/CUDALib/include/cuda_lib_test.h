#pragma once
#include "cuda_runtime.h"
#include "vector_types.h" 
#include "vector_functions.h" 
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#ifdef __INTELLISENSE__
#define CUDA_SINGLE_LOOP(grid, block) SingleLoop <<< (grid), (block) >>>
#define KERNEL_ARGS2(grid, block) <<< (grid), (block) >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

void CudaSingleLoop(dim3 &grid, dim3 &block);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, std::string *error_message);
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda2(int4 *c, const int4 *a, const int4 *b, std::string* error_message);