#pragma once

// the fork's handover of tensor ranges from the thread that allocates the backend buffers to the thread that loads
// the data (llama_model_base::load_tensors), included by llama-model-loader.h

#include "ggml.h"

#include <condition_variable>
#include <deque>
#include <mutex>

// tensors sharing one freshly allocated backend buffer, last == nullptr means the end of the context
struct llama_tensor_range {
    ggml_tensor * first = nullptr;
    ggml_tensor * last  = nullptr;
};

// hands ranges from the thread allocating the backend buffers to the thread loading the tensor
// data, so that reading from the file overlaps the driver allocating the next buffer.
// single producer, single consumer
struct llama_tensor_range_queue {
    void push(ggml_tensor * first, ggml_tensor * last) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ranges.push_back({ first, last });
        }
        cv.notify_all();
    }

    // no more ranges will be pushed
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }
        cv.notify_all();
    }

    // close and wait until the consumer is done with the tensors handed over so far, for when the
    // buffers holding them are about to be freed.
    // a consumer that has not entered yet is not waited for: the queue is emptied and closed here,
    // so its pop() returns false and it never learns of a tensor. this holds only as long as the
    // consumer touches nothing but what pop() gave it.
    void drain() {
        std::unique_lock<std::mutex> lock(mutex);
        ranges.clear();
        closed = true;
        cv.notify_all();
        cv.wait(lock, [this] { return !active; });
    }

    // blocks until a range arrives, false once the queue is closed and drained
    bool pop(llama_tensor_range & range) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !ranges.empty() || closed; });
        if (ranges.empty()) {
            return false;
        }
        range = ranges.front();
        ranges.pop_front();
        return true;
    }

    void enter() {
        std::lock_guard<std::mutex> lock(mutex);
        active = true;
    }

    void leave() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            active = false;
        }
        cv.notify_all();
    }

private:
    std::mutex                      mutex;
    std::condition_variable         cv;
    std::deque<llama_tensor_range>  ranges;
    bool                            closed = false;
    bool                            active = false;
};
