// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "cuda_lib_test.h"
#include "GameFramework/Actor.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"
#include "CUDA_Test.generated.h"


UCLASS()
class CUDATEST_API ACUDA_Test : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACUDA_Test();

	
	UFUNCTION(BlueprintCallable, Category = "CUDATest")
	bool SimpleCUDATest();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	
	
};
