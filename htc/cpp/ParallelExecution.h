// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#pragma once

#include <thread>
#include <deque>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

class ParallelExecution
{
public:
    /**
     * @brief Provides simple methods to parallelize for loops including helper functions for critical sections.
     * 
     * @param numbThreads specifies the number of threads used for parallelization (if not specified otherwise). Defaults to the number of cores available on the system (virtual + real cores)
     */
    explicit ParallelExecution(const size_t numbThreads = std::thread::hardware_concurrency())
        : numbThreads(numbThreads)
    {
        assert(numbThreads > 0 && "At least one thread is necessary");
    }

    /**
     * @brief Stores results in a thread-safe way.
     * 
     * A mutex will automatically be locked on entry and unlocked on exit of this function. This is useful after the parallel computation when a common variable is accessed containing all the results.
     * 
     * @param callback includes the code which should be executed in a thread-safe way
     */
    void setResult(const std::function<void()>& callback)
    {
        std::lock_guard<std::mutex> lock(mutexResult);  // Only one thread at a time is allowed to change the result value (unlocks automatically on destructor call)

        callback();
    }

    /**
     * @brief Writes messages to the console in a thread-safe way.
     * 
     * Same mutex behaviour as in ParallelExecution::setResult(). Useful if you don't want your console output get messed up.
     * 
     * @param message to print to the console
     */
    void write(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(mutexConsole);

        std::cout << message << std::endl;
    }

    /**
     * @brief Executes index-based containers in parallel.
     * 
     * It is save to throw exceptions from inside the threads. They are catched and re-thrown later in the main thread.
     * 
     * @param idxBegin first index to start (including), e.g. <code>0</code>
     * @param idxEnd last index to start (including), e.g. <code>container.size()</code>
     * @param callback this function will be called from each thread multiple times. Each time an associated index will be passed to the function
     * @param numbThreadsFor number of threads which should be used for parallelisation (only for the current loop)
     */
    void parallel_for(const size_t idxBegin, const size_t idxEnd, const std::function<void(const size_t)>& callback, const size_t numbThreadsFor = CLASS_SETTING)
    {
        /* If the user provides a thread number for this loop, use it. Otherwise, use the thread number stored in the class variable */
        const size_t _numbThreadsFor = numbThreadsFor == CLASS_SETTING ? numbThreads : numbThreadsFor;
        const size_t sizeThreads = std::min(idxEnd - idxBegin + 1, _numbThreadsFor);
        assert(sizeThreads > 0 && "No index range given");

        const auto threadFunction = [&callback, this] (const size_t begin, const size_t end)
        {
            // Catch any exceptions, store them and re-throw them later in the main thread (https://stackoverflow.com/questions/2209224/vector-vs-list-in-stl)
            try
            {
                for (size_t i = begin; i <= end; ++i)
                {
                    callback(i);    // The thread function executes the callback for every assigned index
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mutexExceptions);
                threadExceptions.push_back(std::current_exception());
            }
        };

        if (sizeThreads == 1)       // There is no need to parallelize when only one idx is given, just execute in the main thread
        {
            threadFunction(idxBegin, idxEnd);
            checkExceptions();
            return;
        }

        std::deque<std::thread> threads(sizeThreads);
        
        /* Calculate the index ranges */
        const size_t n = idxEnd - idxBegin + 1;    // Both are inclusive
        const size_t nEqual = n / sizeThreads;     // 38 / 12 = 3
        const size_t nRest = n % sizeThreads;      // 38 % 12 = 2
        size_t d = 0;                              // The last part should be portioned equally between all threads

        /*
        
        # Thread 0
        d = 0 -> d = 1
        0*3, ..., 1*3-1 (+1)
        0, 1, 2, 3

        # Thread 1
        d = 1 -> d = 2
        1*3 (+1), ..., 2*3-1 (+2)
        4, 5, 6, 5, 7

        # Thread 2
        d = 2 -> d = 2
        2*3 (+2), ..., 3*3-1 (+2)
        8, 9, 10

         */

         /* Start every thread */
        for (size_t tid = 0; tid < threads.size(); tid++)
        {
            const size_t dNew = d < nRest ? d + 1 : d;
            threads[tid] = std::thread(threadFunction, idxBegin + tid * nEqual + d, idxBegin + (tid + 1)*nEqual - 1 + dNew);  // Execute the threadFunction which calls the callback from the user
            d = dNew;
        }

        /* And wait until they are finished with calculation */
        for (size_t i = 0; i < threads.size(); i++)
        {
            threads[i].join();
        }

        checkExceptions();
    }

    std::deque<std::exception_ptr>& getThreadExceptions()
    {
        return threadExceptions;
    }

private:
    void checkExceptions()
    {
        for (const std::exception_ptr exception : threadExceptions)
        {
            std::rethrow_exception(exception);
        }
    }

private:
    size_t numbThreads;
    std::mutex mutexResult;
    std::mutex mutexConsole;
    static const size_t CLASS_SETTING = 0;

    std::mutex mutexExceptions;
    std::deque<std::exception_ptr> threadExceptions;
};
