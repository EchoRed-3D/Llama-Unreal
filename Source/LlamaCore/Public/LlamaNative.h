// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaThreading.h"
#include "CoreMinimal.h"


/** 
* C++ native wrapper in Unreal styling for Llama.cpp with threading and callbacks. Embed in final place
* where it should be used e.g. ActorComponent, UObject, or Subsystem subclass.
*/
class LLAMACORE_API FLlamaNative : public FLlamaThreading
{
public:

	//Callbacks
	TFunction<void(const FString& Token)> OnTokenGenerated;
	TFunction<void(const FString& Partial)> OnPartialGenerated;		//usually considered sentences, good for TTS.
	TFunction<void(const FString& Response)> OnResponseGenerated;	//per round
	TFunction<void(int32 TokensProcessed, EChatTemplateRole ForRole, float Speed)> OnPromptProcessed;	//when an inserted prompt has finished processing (non-generation prompt)
	TFunction<void()> OnGenerationStarted;
	TFunction<void(const FLlamaRunTimings& Timings)> OnGenerationFinished;
	TFunction<void(const FString& ErrorMessage, int32 ErrorCode)> OnError;
	TFunction<void(const FLLMModelState& UpdatedModelState)> OnModelStateChanged;

	//Expected to be set before load model
	void SetModelParams(const FLLMModelParams& Params);

	//Loads the model found at ModelParams.PathToModel, use SetModelParams to specify params before loading
	void LoadModel(bool bForceReload = false, TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback = nullptr);
	bool IsModelLoaded();

	//Prompt input
	void InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	void InsertRawPrompt(const FString& Prompt, bool bGenerateReply = true, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	bool IsGenerating();
	void StopGeneration();
	void ResumeGeneration();

	//Context change - not yet implemented
	void ResetContextHistory(bool bKeepSystemPrompt = false);	//full reset
	void RemoveLastUserInput();		//chat rollback to undo last user input
	void RemoveLastReply();		//chat rollback to undo last assistant input.
	void RegenerateLastReply(); //removes last reply and regenerates (changing seed?)

	//Base api to do message rollback
	void RemoveLastNMessages(int32 MessageCount);	//rollback

	//Pure query of current game thread context
	void SyncPassedModelStateToNative(FLLMModelState& StateToSync);

	FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate, bool bAddAssistantBoS = false);

	FLlamaNative();
	~FLlamaNative();


protected:

	//can be safely called on game thread or the bg thread
	void SyncModelStateToInternal(TFunction<void()>AdditionalGTStateUpdates = nullptr);

	//utility functions, only safe to call on bg thread
	int32 RawContextHistory(FString& OutContextString);
	void GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory);
	int32 UsedContextLength();

	//GT State - safely accesible on game thread
	FLLMModelParams ModelParams;
	FLLMModelState ModelState;

	//BG State - do not read/write on GT
	FString CombinedPieceText;	//accumulates tokens into full string during per-token inference.


	class FLlamaInternal* Internal = nullptr;
	
};