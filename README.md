# CUDA in Unreal Engine 4

An implementation of [SCIEMENT](http://www.sciement.com/tech-blog/author/sciement/)'s post on how to use CUDA functions with Unreal Engine 4 at [`[C++][CUDA][UE4]CUDA関数をUnreal Engine 4で用いる`](http://www.sciement.com/tech-blog/c/cuda_in_ue4/) 

## Development Environment

    Visual Studio 2015

    CUDA Toolkit v10.1

    Unreal Engine 4.18


## How it works
To use CUDA functions in UE4 we need to first implement cuda functions and build it into a lib (cuda_lib/) and then link it to Unreal project. Here we implement 2 versions of parallel add function to add two vectors. The first one works with oridianry C-based array while the second one uses CUDA specified vector (int4) as its parameters.

## Custom Library (cuda_lib/)
### Implement needed functions in **cuda_lib** project:

cuda_lib_test.h
```cpp
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, std::string *error_message);
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda2(int4 *c, const int4 *a, const int4 *b, std::string* error_message);
```

### Implement addition:

cuda_lib_test.cu

```cpp
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
```

### Allocate GPU buffer:

cuda_lib_test.cu (**addWithCuda()**)
```cpp
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
		*error_message = "cudaMalloc failed!";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
		*error_message = "cudaMalloc failed!";
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
		*error_message = "cudaMalloc failed!";
        goto Error;
    }
```

### Copy data to GPU buffer
cuda_lib_test.cu (**addWithCuda()**)
```cpp
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		*error_message = "cudaMemcpy failed!";
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		*error_message = "cudaMemcpy failed!";
        goto Error;
    }
```
### Compute with GPU
```cpp
    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size>>>(dev_c, dev_a, dev_b);
```

## UE4 Test Project (CUDATest/)

### Link libraries
CUDATest.Build.cs
```C#
    string custom_cuda_lib_include = "CUDALib/include";
    string custom_cuda_lib_lib = "CUDALib/lib";

    PublicIncludePaths.Add(Path.Combine(poject_root_path, custom_cuda_lib_include));
    PublicAdditionalLibraries.Add(Path.Combine(poject_root_path, custom_cuda_lib_lib, "cuda_lib.lib"));

    string cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/";
    string cuda_include = "include";
    string cuda_lib = "lib/x64";

    PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

    //PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart.lib"));
    PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));
```

### Specify function in C++ class
CUDA_Test.h
```cpp
    UFUNCTION(BlueprintCallable, Category = "CUDATest")
	bool SimpleCUDATest();
```
### Use custom CUDA function
CUDA_Test.cpp (**SimpleCUDATest()**)
```cpp
// ----- addWithCuda test -----
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };
	std::string error_message;

	// Add vectors in parallel.
	cudaError_t cuda_status = addWithCuda(c, a, b, arraySize, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("addWithCuda failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	UE_LOG(LogTemp, Warning, TEXT("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}"), c[0], c[1], c[2], c[3], c[4]);
```

### Call test function in Blueprint
![](Docs/blueprint.png)

### Execution result
![](Docs/outputlog.png)
