// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "common/Common.h"
#include "llama.h"

#include <algorithm>
#include <fstream>
#include <iostream>

struct chunk {
    // filename
    std::string filename;
    // original file position
    size_t filepos;
    // original text data
    std::string textdata;
    // tokenized text data
    std::vector<llama_token> tokens;
    // embedding
    std::vector<float> embedding;
};

/**
 * 
 */
class LLAMACORE_API LlamaRetrieval
{
public:

	//Core State
	llama_model* LlamaModel = nullptr;
	llama_context* Context = nullptr;
    common_params params;


    void BuildVectorDataBase(struct FLLMRetrivalParams Params, std::vector<std::string> context_files);
    void Unload();

    FString Query(FString Query);


    LlamaRetrieval();
    ~LlamaRetrieval();

    // Step 1 Loader And Splitter
    // Step 2 Embedding
    // Step 3 Vector Store
    // Step 4 Retrieval : Query

protected:

    std::vector<chunk> chunk_files(std::vector<std::string> context_files, int chunk_size, const std::string& chunk_separator);
    std::vector<chunk> chunk_file(const std::string& filename, int chunk_size, const std::string& chunk_separator);

    void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
    void batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int nb_embd);



    std::vector<chunk> chunks;


};
