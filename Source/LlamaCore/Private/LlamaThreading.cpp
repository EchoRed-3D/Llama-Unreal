// Fill out your copyright notice in the Description page of Project Settings.


#include "LlamaThreading.h"
#include "LlamaUtility.h"

#include "Async/TaskGraphInterfaces.h"
#include "Async/Async.h"
#include "Tickable.h"

FLlamaThreading::FLlamaThreading()
{
}

FLlamaThreading::~FLlamaThreading()
{
    bThreadShouldRun = false;

    //Remove ticker if active
    RemoveTicker();

    //Wait for the thread to stop
    while (bThreadIsActive)
    {
        FPlatformProcess::Sleep(0.01f);
    }
}

void FLlamaThreading::ClearPendingTasks(bool bClearGameThreadCallbacks)
{
    BackgroundTasks.Empty();

    if (bClearGameThreadCallbacks)
    {
        GameThreadTasks.Empty();
    }
}

void FLlamaThreading::OnGameThreadTick(float DeltaTime)
{
    //Handle all the game thread callbacks
    if (!GameThreadTasks.IsEmpty())
    {
        //Run all queued tasks
        while (!GameThreadTasks.IsEmpty())
        {
            FLLMThreadTask Task;
            GameThreadTasks.Dequeue(Task);
            if (Task.TaskFunction)
            {
                //Run Task
                Task.TaskFunction(Task.TaskId);
            }
        }
    }
}

void FLlamaThreading::AddTicker()
{
    TickDelegateHandle = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([this](float DeltaTime)
        {
            OnGameThreadTick(DeltaTime);
            return true;
        }));
}

void FLlamaThreading::RemoveTicker()
{
    if (IsNativeTickerActive())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickDelegateHandle);
        TickDelegateHandle = nullptr;
    }
}

bool FLlamaThreading::IsNativeTickerActive()
{
    return TickDelegateHandle.IsValid();
}

void FLlamaThreading::StartLLMThread()
{
    bThreadShouldRun = true;
    Async(EAsyncExecution::Thread, [this]
        {
            bThreadIsActive = true;

            while (bThreadShouldRun)
            {
                //Run all queued tasks
                while (!BackgroundTasks.IsEmpty())
                {
                    FLLMThreadTask Task;
                    BackgroundTasks.Dequeue(Task);
                    if (Task.TaskFunction)
                    {
                        //Run Task
                        Task.TaskFunction(Task.TaskId);
                    }
                }

                FPlatformProcess::Sleep(ThreadIdleSleepDuration);
            }

            bThreadIsActive = false;
        });
}

int64 FLlamaThreading::GetNextTaskId()
{
    //technically returns an int32
    return TaskIdCounter.Increment();
}

void FLlamaThreading::EnqueueBGTask(TFunction<void(int64)> TaskFunction)
{
    //Lazy start the thread on first enqueue
    if (!bThreadIsActive)
    {
        StartLLMThread();
    }

    FLLMThreadTask Task;
    Task.TaskId = GetNextTaskId();
    Task.TaskFunction = TaskFunction;

    BackgroundTasks.Enqueue(Task);
}

void FLlamaThreading::EnqueueGTTask(TFunction<void()> TaskFunction, int64 LinkedTaskId)
{
    FLLMThreadTask Task;

    if (LinkedTaskId == -1)
    {
        Task.TaskId = GetNextTaskId();
    }
    else
    {
        Task.TaskId = LinkedTaskId;
    }

    Task.TaskFunction = [TaskFunction](int64 InTaskId)
        {
            TaskFunction();
        };

    GameThreadTasks.Enqueue(Task);
}
