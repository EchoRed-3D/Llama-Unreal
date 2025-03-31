// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "LlamaThreading.h"
#include "CoreMinimal.h"

/**
 * 
 */
class LLAMACORE_API FLlamaNativeRetrieval : public FLlamaThreading
{
public:

	//Callbacks
	TFunction<void(const FLLMModelState& UpdatedModelState)> OnModelStateChanged;
	TFunction<void(float PercentProgress, int32 nTokens, int32 nSequence)> OnVectorStoreProgress;


	FLlamaNativeRetrieval();
	~FLlamaNativeRetrieval();

	//Expected to be set before load model
	void SetRetrievalParams(const FLLMRetrivalParams& Params);
	void SetContextFiles(const std::vector<std::string> context_files);

	//Loads the model found at RetrivalParams.PathToModel, use SetRetrievalParams to specify params before loading
	void LoadModel(bool bForceReload = false, TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback = nullptr);
	bool IsModelLoaded();


	void CreateVectorStore(TFunction<void(FLLMVectorStore VectorStore, int32 StatusCode)> ModelVectorStoreCallback = nullptr);
	void QueryVectorSore(FLLMVectorStore VectorStore, FString Query,
		TFunction<void(TArray<FLLMQueryReponse> QueryReponse, int32 StatusCode)> ModelQueryCallback = nullptr);

protected:

	FLLMRetrivalParams RetrivalParams;
	FLLMModelState ModelState;
	std::vector<std::string> Context_files;

	class FLlamaRetrieval* Retrieval = nullptr;
};
