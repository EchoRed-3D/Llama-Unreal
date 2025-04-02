// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "LlamaDataTypes.h"
#include "common/Common.h"
#include "llama.h"

#include <algorithm>
#include <fstream>
#include <iostream>


struct chunk_params
{
    // chunk size for context embedding
    int chunk_size = 256;
   
    // chunk separator for context embedding
    std::string chunk_separator = "\n"; 

    //character overlap between chunks
    int chunk_overlap = 100;

};


/**
 * 
 */
class LLAMACORE_API FLlamaRetrieval
{
public:

	//Core State
	llama_model* LlamaModel = nullptr;
	llama_context* Context = nullptr;
    common_params params;
    chunk_params ChunkParams;

    //main streaming callback
    TFunction<void(float PercentProgress, int32 nTokens, int32 nSequence)>OnVectorStoreProgress = nullptr;


    //Model loading
    bool LoadModel(struct FLLMRetrivalParams Params);
    void UnloadModel();
    bool IsModelLoaded();


    std::vector<chunk> CreateVectorStore(std::vector<std::string> context_files);
    // Step 1 Loader And Splitter
    std::vector<chunk> SplitterFiles(std::vector<std::string> context_files, chunk_params c_params);
    // Step 2 Embedding
    bool EmbeddingFiles(std::vector<chunk>& ChunkFiles);

    // Step 3 Retrieval : Query
    TArray<FLLMQueryReponse> QueryVectorStore(std::vector <chunk> VectorStore, FString Query);

    FLlamaRetrieval();
    ~FLlamaRetrieval();


protected:

    std::vector<chunk> chunk_file(const std::string& filename, chunk_params c_params);

    void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
    void batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int nb_embd);

    //std::vector<chunk> chunks;


    FThreadSafeBool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;
};
