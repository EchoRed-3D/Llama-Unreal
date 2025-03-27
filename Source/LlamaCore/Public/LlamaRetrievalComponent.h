// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "LlamaDataTypes.h"


#include "LlamaRetrievalComponent.generated.h"


UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaRetrievalComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	ULlamaRetrievalComponent(const FObjectInitializer& ObjectInitializer);
	// Sets default values for this component's properties
	ULlamaRetrievalComponent();
	~ULlamaRetrievalComponent();


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM  Retrival Component")
	FLLMRetrivalParams RetrivalParams;


	//UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM  Retrival Component")
	//TArray<FString> ContextFiles;

	UFUNCTION(BlueprintCallable, Category = "LLM Retrival Component")
	void TryBuildVectorDataBase(TArray<FString> ContextFiles);

	UFUNCTION(BlueprintCallable, Category = "LLM Retrival Component")
	void Unload();

	UFUNCTION(BlueprintCallable, Category = "LLM Retrival Component")
	FString Query(FString Query);

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	

private:
	class LlamaRetrieval* Retrieval;
};
