#pragma once

// the fork's helpers of common.h (common.h includes this header)

#include <cstddef>

// Host (system) RAM introspection. Returns 0 when the value cannot be determined.
// Unlike ggml's CPU device query, these report the real available memory on all platforms.
size_t common_host_mem_total();     // total physical RAM in bytes
size_t common_host_mem_available(); // currently available/free RAM in bytes
