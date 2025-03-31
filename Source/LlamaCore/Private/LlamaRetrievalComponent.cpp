// Fill out your copyright notice in the Description page of Project Settings.


#include "LlamaRetrievalComponent.h"
#include "LlamaNativeRetrieval.h"
#include "LlamaUtility.h"

ULlamaRetrievalComponent::ULlamaRetrievalComponent(const FObjectInitializer& ObjectInitializer) : UActorComponent(ObjectInitializer)
{
	NativeRetrieval = new FLlamaNativeRetrieval();

	//Hookup native callbacks
	NativeRetrieval->OnModelStateChanged = [this](const FLLMModelState& UpdatedModelState)
	{
			ModelState = UpdatedModelState;
	};


	NativeRetrieval->OnVectorStoreProgress = [this](float PercentProgress, int32 nTokens, int32 nSequence)
		{
			OnVectorStoreProgress.Broadcast(PercentProgress, nTokens, nSequence);
		};

	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = true;
}

// Sets default values for this component's properties
ULlamaRetrievalComponent::ULlamaRetrievalComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	
	// ...
}

ULlamaRetrievalComponent::~ULlamaRetrievalComponent()
{
	if (NativeRetrieval)
	{
		delete NativeRetrieval;
		NativeRetrieval = nullptr;
	}
}


void ULlamaRetrievalComponent::LoadModel(bool bForceReload)
{
	NativeRetrieval->SetRetrievalParams(RetrivalParams);
	NativeRetrieval->LoadModel(bForceReload, [this](const FString& ModelPath, int32 StatusCode)
	{
		//We errored, the emit will happen before we reach here so just exit
		if (StatusCode != 0)
		{
			return;
		}

		OnModelLoaded.Broadcast(ModelPath);
	});
}


void ULlamaRetrievalComponent::Unload()
{
	NativeRetrieval->UnloadModel([this](int32 StatusCode)
		{
			//this pretty much should never get called, just in case: emit.
			if (StatusCode != 0)
			{
				FString ErrorMessage = FString::Printf(TEXT("UnloadModel returned error code: %d"), StatusCode);
				UE_LOG(LlamaLog, Warning, TEXT("%s"), *ErrorMessage);				
			}
		});
}

void ULlamaRetrievalComponent::CreateVectorStore(TArray<FString> ContextFiles)
{
	if (ContextFiles.IsEmpty())
	{
		UE_LOG(LlamaLog, Warning, TEXT("ContextFiles is Empty"))
		return;
	}


	std::vector<std::string> context_files;
	for (const auto File : ContextFiles)
	{
		context_files.push_back(FLlamaString::ToStd(File));
	}

	NativeRetrieval->SetContextFiles(context_files);
	NativeRetrieval->CreateVectorStore([this](FLLMVectorStore VectorStore, int32 StatusCode)
		{
			if (StatusCode != 0)
			{
				return;
			}

			OnVectorStoreCreated.Broadcast(VectorStore);

		});
}

void ULlamaRetrievalComponent::QueryVectorStore(FLLMVectorStore VectorStore, FString Query)
{
	if (!VectorStore.IsValid())
	{
		UE_LOG(LlamaLog, Warning, TEXT("Invalid Vector Store"));
		return;
	}

	NativeRetrieval->QueryVectorSore(VectorStore, Query, [this](TArray<FLLMQueryReponse> QueryReponse, int32 StatusCode) 
		{
			if (StatusCode != 0)
			{
				return;
			}

			OnQueryReponses.Broadcast(QueryReponse);

		});
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


	NativeRetrieval->OnGameThreadTick(DeltaTime);
	// ...
}

