// Fill out your copyright notice in the Description page of Project Settings.


#include "LlamaRetrievalComponent.h"
#include "Internal/LlamaRetrieval.h"
#include "LlamaUtility.h"

ULlamaRetrievalComponent::ULlamaRetrievalComponent(const FObjectInitializer& ObjectInitializer) : UActorComponent(ObjectInitializer)
{
	Retrieval = new LlamaRetrieval();
}

// Sets default values for this component's properties
ULlamaRetrievalComponent::ULlamaRetrievalComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}

ULlamaRetrievalComponent::~ULlamaRetrievalComponent()
{
	if (Retrieval)
	{
		delete Retrieval;
		Retrieval = nullptr;
	}
}


void ULlamaRetrievalComponent::TryBuildVectorDataBase(TArray<FString> ContextFiles)
{
	if (ContextFiles.IsEmpty())
	{
		UE_LOG(LlamaLog,Warning,TEXT("ContextFiles is Empty"))
		return;
	}


	std::vector<std::string> context_files;
	for (const auto File : ContextFiles)
	{
		context_files.push_back(FLlamaString::ToStd(File));
	}

	Retrieval->BuildVectorDataBase(RetrivalParams, context_files);
}

void ULlamaRetrievalComponent::Unload()
{
	Retrieval->Unload();
}

FString ULlamaRetrievalComponent::Query(FString Query)
{
	return Retrieval->Query(Query);
}

// Called when the game starts
void ULlamaRetrievalComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void ULlamaRetrievalComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

