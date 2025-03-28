// Fill out your copyright notice in the Description page of Project Settings.


#include "Internal/LlamaRetrieval.h"
#include "LlamaUtility.h"
#include "LlamaDataTypes.h"
#include "common/common.h"
#include "common/sampling.h"

void LlamaRetrieval::BuildVectorDataBase(struct FLLMRetrivalParams Params, std::vector<std::string> context_files)
{

    common_init();

    // load dynamic backends
    ggml_backend_load_all();

    std::string Path = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(Params.PathToModel));

    
    params.n_batch = Params.Nbatch;
    params.n_ubatch = Params.Nbatch;
    params.n_ctx = Params.MaxContextLength;
    params.model = Path.c_str();
    params.embedding = true;
    params.chunk_size = Params.ChunkSize;
    params.chunk_separator = FLlamaString::ToStd(Params.ChunkSeparator);
    params.sampling.top_k = Params.TopK;

    //Files for embedding
    params.context_files = context_files;



    if (params.chunk_size <= 0) {
        UE_LOG(LlamaLog, Warning, TEXT("chunk_size must be positive"));
        return;
    }
    if (params.context_files.empty()) {
        UE_LOG(LlamaLog, Warning, TEXT("context_files must be specified"));
        return;
    }

    UE_LOG(LlamaLog, Display, TEXT("cprocessing files:"));
    for (auto& context_file : params.context_files) 
    {
        UE_LOG(LlamaLog, Verbose, TEXT("%hs"), context_file.c_str());
    }

    // Build ChunkFiles Slicer
    chunks = chunk_files(params.context_files, params.chunk_size, params.chunk_separator);

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    LlamaModel = llama_init.model.get();
    Context = llama_init.context.get();

   

    if (LlamaModel == NULL)
    {
        UE_LOG(LlamaLog, Warning, TEXT("unable to load mode"));
        return;
    }

    UE_LOG(LlamaLog, Warning, TEXT("Model Loaded"));


    const llama_vocab* vocab = llama_model_get_vocab(LlamaModel);

    const int n_ctx_train = llama_model_n_ctx_train(LlamaModel);
    const int n_ctx = llama_n_ctx(Context);

    const enum llama_pooling_type pooling_type = llama_pooling_type(Context);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) 
    {
        UE_LOG(LlamaLog, Error, TEXT("%hs: pooling type NONE not supported"), __func__);
        return;
    }

    if (n_ctx > n_ctx_train)
    {
        UE_LOG(LlamaLog, Warning, TEXT("%hs: warning: model was trained on only %d context tokens (%d specified)"), __func__, n_ctx_train, n_ctx);
    }

    // Print system information
    UE_LOG(LlamaLog, Display, TEXT("%hs"), common_params_get_system_info(params).c_str());


    // max batch size
    const uint64_t n_batch = params.n_batch;
    //GGML_ASSERT(params.n_batch >= params.n_ctx);

    if (chunks.size() <= 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Chunks is Empty"));
        return;
    }
    // tokenize the prompts and trim
    for (auto& chunk : chunks) 
    {
        auto inp = common_tokenize(Context, chunk.textdata, true, false);
        if (inp.size() > n_batch)
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run"),
                __func__, (long long int) inp.size(), (long long int) n_batch);
            return;
        }
        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab)))
        {
            inp.push_back(llama_vocab_eos(vocab));
        }
        chunk.tokens = inp;
    }

    // tokenization stats
    if (params.verbose_prompt) 
    {
        for (int i = 0; i < (int)chunks.size(); i++) 
        {
            UE_LOG(LlamaLog, Display, TEXT("%hs: prompt %d: '%hs'"), __func__, i, chunks[i].textdata.c_str());
            UE_LOG(LlamaLog, Display, TEXT("%hs: number of tokens in prompt = %zu"), __func__, chunks[i].tokens.size());

            for (int j = 0; j < (int)chunks[i].tokens.size(); j++) 
            {
                UE_LOG(LlamaLog, Display, TEXT("-> '%hs'"), common_token_to_piece(Context, chunks[i].tokens[j]).c_str());
            }
        }
    }


    // initialize batch
    const int n_chunks = chunks.size();
    params.n_chunks = n_chunks;
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_model_n_embd(LlamaModel);
    std::vector<float> embeddings(n_chunks * n_embd, 0);
    float* emb = embeddings.data();


    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) {
        // clamp to n_batch tokens
        auto& inp = chunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float* out = emb + p * n_embd;
            batch_decode(Context, batch, out, s, n_embd);
            common_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }


    // final batch
    float* out = emb + p * n_embd;
    batch_decode(Context, batch, out, s, n_embd);

    // save embeddings to chunks
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embedding = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
        // clear tokens as they are no longer needed
        chunks[i].tokens.clear();
    }

    Query(FString());
    
}

void LlamaRetrieval::Unload()
{
    if (Context)
    {
        //llama_free(Context);
        UE_LOG(LlamaLog, Display, TEXT("Unload Context"))

       Context = nullptr;
    }
    if (LlamaModel)
    {
        //llama_model_free(LlamaModel);
        UE_LOG(LlamaLog, Display, TEXT("Unload Model"))

        LlamaModel = nullptr;
    }

    llama_backend_free();
}

FString LlamaRetrieval::Query(FString Query)
{
    if (!Context)
    {
        UE_LOG(LlamaLog, Display, TEXT("%hs Context Invalid"),__func__)
        return FString();
    }



    // max batch size
    const uint64_t n_batch = llama_n_batch(Context);
    UE_LOG(LlamaLog, Display, TEXT("Query n_batch:%d ctx_n_batch:%d"), params.n_batch, n_batch)

    // initialize batch
    const int n_chunks = chunks.size();
    // allocate output
    const int n_embd = llama_model_n_embd(LlamaModel);
    // Start Query
    llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
  

    //std::string query = FLlamaString::ToStd(Query);
    std::string query = "Quelle est la meilleure formule pour un shampoing sur des cheveux gras ?";
    UE_LOG(LlamaLog, Display, TEXT("Enyer Query : %S"), query.c_str())
     
    //std::getline(std::cin, query);

    std::vector<llama_token> query_tokens = common_tokenize(Context, query, true);

    batch_add_seq(query_batch, query_tokens, 0);

    std::vector<float> query_emb(n_embd, 0);
    batch_decode(Context, query_batch, query_emb.data(), 1, n_embd);

    common_batch_clear(query_batch);

    // compute cosine similarities
    std::vector<std::pair<int, float>> similarities;
    for (int i = 0; i < n_chunks; i++) {
        float sim = common_embd_similarity_cos(chunks[i].embedding.data(), query_emb.data(), n_embd);
        similarities.push_back(std::make_pair(i, sim));
    }

    // sort similarities
    std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
        });

    UE_LOG(LlamaLog, Display, TEXT("Top %d similar chunks:"), params.sampling.top_k)
        for (int i = 0; i < std::min(params.sampling.top_k, (int)chunks.size()); i++)
        {
            UE_LOG(LlamaLog, Display, TEXT("filename: %hs"), chunks[similarities[i].first].filename.c_str());
            UE_LOG(LlamaLog, Display, TEXT("filepos: %lld"), (long long int) chunks[similarities[i].first].filepos);
            UE_LOG(LlamaLog, Display, TEXT("similarity: %f"), similarities[i].second);

            FString Text = FLlamaString::ToUE((chunks[similarities[i].first].textdata));
            UE_LOG(LlamaLog, Display, TEXT("textdata:\n%s"), *Text);
            UE_LOG(LlamaLog, Display, TEXT("--------------------"))
        }

    //llama_perf_context_print(Context);
    //llama_kv_cache_clear(Context);
    llama_batch_free(query_batch);
    return FString();
}

LlamaRetrieval::LlamaRetrieval()
{
}

LlamaRetrieval::~LlamaRetrieval()
{
    //llama_perf_context_print(Context);

    Unload();

    //llama_batch_free(query_batch);
    llama_backend_free();
}

std::vector<chunk> LlamaRetrieval::chunk_files(std::vector<std::string> context_files, int chunk_size, const std::string& chunk_separator)
{
    std::vector<chunk> ChunksFiles;

    for (auto& context_file : context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, chunk_size, chunk_separator);
        ChunksFiles.insert(ChunksFiles.end(), file_chunk.begin(), file_chunk.end());
    }
    UE_LOG(LlamaLog, Display, TEXT("Number of chunks: %zu"), ChunksFiles.size());

    return ChunksFiles;
}

std::vector<chunk> LlamaRetrieval::chunk_file(const std::string& filename, int chunk_size, const std::string& chunk_separator)
{
	std::vector<chunk> chunksFile;
	std::ifstream f(filename.c_str());

	if (!f.is_open()) 
	{
		UE_LOG(LlamaLog,Warning, TEXT("could not open file %hs"), filename.c_str());
		return chunksFile;
	}


    chunk current_chunk;
    char buffer[1024];
    int64_t filepos = 0;
    std::string current;
    while (f.read(buffer, 1024)) {
        current += std::string(buffer, f.gcount());
        size_t pos;
        while ((pos = current.find(chunk_separator)) != std::string::npos) {
            current_chunk.textdata += current.substr(0, pos + chunk_separator.size());
            if ((int)current_chunk.textdata.size() > chunk_size) {
                // save chunk
                current_chunk.filepos = filepos;
                current_chunk.filename = filename;
                chunksFile.push_back(current_chunk);
                // update filepos
                filepos += (int)current_chunk.textdata.size();
                // reset current_chunk
                current_chunk = chunk();
            }
            current = current.substr(pos + chunk_separator.size());
        }

    }
    // add leftover data to last chunk
    if (current_chunk.textdata.size() > 0) {
        if (chunksFile.empty()) {
            current_chunk.filepos = filepos;
            current_chunk.filename = filename;
            chunksFile.push_back(current_chunk);
        }
        else {
            chunksFile.back().textdata += current_chunk.textdata;
        }
    }
    f.close();
    return chunksFile;
}

void LlamaRetrieval::batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id)
{
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) 
    {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

void LlamaRetrieval::batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int nb_embd)
{
    if (!ctx) { UE_LOG(LlamaLog, Error, TEXT("Contex Invalid")) return; }

    llama_kv_self_clear(ctx);


    // run model
    UE_LOG(LlamaLog, Verbose, TEXT("%hs: n_tokens = %d, n_seq = %d"), __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) 
    {
        UE_LOG(LlamaLog, Warning, TEXT("%hs : failed to decode"), __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float* embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) 
            {
                UE_LOG(LlamaLog, Warning, TEXT("%hs: failed to get embeddings for token %d"), __func__, i);
                continue;
            }
        }

        float* out = output + batch.seq_id[i][0] * nb_embd;
        common_embd_normalize(embd, out, nb_embd, 2);
    }
}
