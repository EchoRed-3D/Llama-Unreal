// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "LlamaDataTypes.h"
#include "CoreMinimal.h"

/**
* C++ native wrapper in Unreal styling for Llama.cpp with threading and callbacks. Embed in final place
* where it should be used e.g. ActorComponent, UObject, or Subsystem subclass.
*/
class LLAMACORE_API FLlamaThreading
{
public:

	FLlamaThreading();
	~FLlamaThreading();

	//if you've queued up a lot of BG tasks, you can clear the queue with this call
	void ClearPendingTasks(bool bClearGameThreadCallbacks = false);

	//tick forward for safely consuming game thread messages
	void OnGameThreadTick(float DeltaTime);
	void AddTicker();	 //optional call this once if you don't forward ticks from e.g. component/actor tick
	void RemoveTicker(); //if you use AddTicker, use remove ticker to balance on exit. Will happen on destruction of FLlamaNative if not called earlier.
	bool IsNativeTickerActive();

	float ThreadIdleSleepDuration = 0.005f;        //default sleep timer for BG thread in sec.

protected:

	//Threading
	void StartLLMThread();
	TQueue<FLLMThreadTask> BackgroundTasks;
	TQueue<FLLMThreadTask> GameThreadTasks;
	FThreadSafeBool bThreadIsActive = false;
	FThreadSafeBool bThreadShouldRun = false;
	FThreadSafeCounter TaskIdCounter = 0;
	int64 GetNextTaskId();

	void EnqueueBGTask(TFunction<void(int64)> Task);
	void EnqueueGTTask(TFunction<void()> Task, int64 LinkedTaskId = -1);

	class FLlamaInternal* Internal = nullptr;
	FTSTicker::FDelegateHandle TickDelegateHandle = nullptr; //optional tick handle - used in subsystem example where tick isn't natively supported
};
