// Fill out your copyright notice in the Description page of Project Settings.

#include "CUDA_Test.h"


// Sets default values
ACUDA_Test::ACUDA_Test()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

inline bool ACUDA_Test::SimpleCUDATest()
{

	for (int i = 0; i < 4; i++)
	{
		UE_LOG(LogTemp, Warning, TEXT("CPU - i = %d\n"), i);
	}
	dim3 grid(1, 1, 1);
	dim3 block(4, 1, 1);
	UE_LOG(LogTemp, Warning, TEXT("-----------\n"));

	//int32 Array[] = { 0, 1, 2, 3 };

	ParallelFor(4, [&](int32 idx_i) {
		ParallelFor(5, [&](int32 idx_j) {
			ParallelFor(3, [&](int32 idx_k) {
				int i = idx_i;
				int j = idx_j;
				int k = idx_k;

				int idx = i + j + k;

				ParallelFor(3, [&](int32 tgt) {
					UE_LOG(LogTemp, Warning, TEXT("(%d, %d, %d), idx = %d, tgt = %d\n"), i, j, k, idx, tgt);
				});
			});
		});
		
	});
	

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

	// ----- addWithCuda2 test -----
	const int4 a_int4 = make_int4(1, 2, 3, 4);
	const int4 b_int4 = make_int4(10, 20, 30, 40);
	int4 c_int4;

	// Add vectors in parallel.
	cuda_status = addWithCuda2(&c_int4, &a_int4, &b_int4, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("addWithCuda failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}
	UE_LOG(LogTemp, Warning, TEXT("{1,2,3,4} + {10,20,30,40} = {%d,%d,%d,%d}"), c_int4.x, c_int4.y, c_int4.z, c_int4.w);

	return true;
}

// Called when the game starts or when spawned
void ACUDA_Test::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ACUDA_Test::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

