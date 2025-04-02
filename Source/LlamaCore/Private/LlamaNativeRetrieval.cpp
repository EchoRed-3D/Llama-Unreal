// Fill out your copyright notice in the Description page of Project Settings.


#include "LlamaNativeRetrieval.h"
#include "Internal/LlamaRetrieval.h"
#include "LlamaUtility.h"

FLlamaNativeRetrieval::FLlamaNativeRetrieval()
{
    Retrieval = new FLlamaRetrieval();

    //TODO Bind callback

    Retrieval->OnVectorStoreProgress = [this](float PercentProgress, int32 nTokens, int32 nSequence)
        {
            if (OnVectorStoreProgress)
            {
                EnqueueGTTask([this, PercentProgress, nTokens, nSequence]()
                    {
                        OnVectorStoreProgress(PercentProgress, nTokens, nSequence);
                    });
            }
        };
}

FLlamaNativeRetrieval::~FLlamaNativeRetrieval()
{
    bThreadShouldRun = false;

    delete Retrieval;
}

void FLlamaNativeRetrieval::SetRetrievalParams(const FLLMRetrivalParams& Params)
{
	RetrivalParams = Params;
}

void FLlamaNativeRetrieval::SetContextFiles(const std::vector<std::string> context_files)
{
    Context_files = context_files;
}

void FLlamaNativeRetrieval::LoadModel(bool bForceReload, TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback)
{
    if (IsModelLoaded() && !bForceReload)
    {
        //already loaded, we're done
        return ModelLoadedCallback(RetrivalParams.PathToEmbeddingModel, 0);
    }

    //Copy so these dont get modified during enqueue op
    const FLLMRetrivalParams ParamsAtLoad = RetrivalParams;


    EnqueueBGTask([this, ParamsAtLoad, ModelLoadedCallback](int64 TaskId)
    {
            //Unload first if any is loaded
            Retrieval->UnloadModel();

            //Now load it
            bool bSuccess = Retrieval->LoadModel(ParamsAtLoad);

            //Sync model state
            if (bSuccess)
            {
                EnqueueGTTask([this, ModelLoadedCallback]
                {
                  ModelState.bModelIsLoaded = true;

                  if (OnModelStateChanged)
                  {
                      OnModelStateChanged(ModelState);
                  }

                  if (ModelLoadedCallback)
                  {
                      ModelLoadedCallback(RetrivalParams.PathToEmbeddingModel, 0);
                  }
                }, TaskId);
            }
            else
            {
                EnqueueGTTask([this, ModelLoadedCallback]
                    {
                        //On error will be triggered earlier in the chain, but forward our model loading error status here
                        ModelLoadedCallback(RetrivalParams.PathToEmbeddingModel, 15);
                    }, TaskId);
            }

    });
}

void FLlamaNativeRetrieval::UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback)
{
    EnqueueBGTask([this, ModelUnloadedCallback](int64 TaskId)
        {
            if (IsModelLoaded())
            {
                Retrieval->UnloadModel();
            }

            //Reply with code
            EnqueueGTTask([this, ModelUnloadedCallback]
                {
                    ModelState.bModelIsLoaded = false;

                    if (OnModelStateChanged)
                    {
                        OnModelStateChanged(ModelState);
                    }

                    if (ModelUnloadedCallback)
                    {
                        ModelUnloadedCallback(0);
                    }
                });
        });
}

bool FLlamaNativeRetrieval::IsModelLoaded()
{
	return ModelState.bModelIsLoaded;
}

void FLlamaNativeRetrieval::CreateVectorStore(TFunction<void(FLLMVectorStore VectorStore, int32 StatusCode)> ModelVectorStoreCallback)
{
    const std::vector<std::string> context_fileAtLoad = Context_files;

    EnqueueBGTask([this, context_fileAtLoad, ModelVectorStoreCallback](int64 TaskId)
        {

            std::vector<chunk> OutVectorStore = Retrieval->CreateVectorStore(context_fileAtLoad);

            if (OutVectorStore.size() > 0)
            {

                EnqueueGTTask([this, OutVectorStore, ModelVectorStoreCallback]
                    {
                        //On error will be triggered earlier in the chain
                        ModelVectorStoreCallback(FLLMVectorStore(OutVectorStore),0);
                    }, TaskId);
            }
            else
            {
                EnqueueGTTask([this, ModelVectorStoreCallback]
                    {
                        //On error will be triggered earlier in the chain
                        ModelVectorStoreCallback(FLLMVectorStore(),15);
                    }, TaskId);
            }
        });
}

void FLlamaNativeRetrieval::QueryVectorSore(FLLMVectorStore VectorStore, FString Query, TFunction<void(TArray<FLLMQueryReponse>QueryReponse, int32 StatusCode)> ModelQueryCallback)
{
    const  std::vector<chunk> VectorStoreAtLoad = VectorStore.GetChunkStd();

    EnqueueBGTask([this, VectorStoreAtLoad, Query, ModelQueryCallback](int64 TaskId)
        {
            TArray<FLLMQueryReponse> QueryReponse = Retrieval->QueryVectorStore(VectorStoreAtLoad, Query);

            if (!QueryReponse.IsEmpty())
            {
                EnqueueGTTask([this, QueryReponse, ModelQueryCallback]
                    {
                        //On error will be triggered earlier in the chain
                        ModelQueryCallback(TArray<FLLMQueryReponse>(QueryReponse), 0);
                    }, TaskId);
            }
            else
            {
                EnqueueGTTask([this, ModelQueryCallback]
                    {
                        //On error will be triggered earlier in the chain
                        ModelQueryCallback(TArray<FLLMQueryReponse>(), 15);
                    }, TaskId);
            }
        });
}
