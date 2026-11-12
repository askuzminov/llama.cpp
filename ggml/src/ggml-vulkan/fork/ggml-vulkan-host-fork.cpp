// fork: what the system says about memory, for the guards of the Vulkan backend (ggml-vulkan-fork.cpp: the device
// memory limit and the elastic KV spill), in a unit of its own so that <windows.h> stays away from vulkan.hpp

#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <pdh.h>
#    include <pdhmsg.h>
#    include <cwchar>
#    include <mutex>
#    include <vector>
#endif

// declared in ggml-vulkan-fork.h, which this unit does not include
uint64_t ggml_vk_fork_host_avail_bytes();
bool     ggml_vk_fork_adapter_dedicated_usage(const uint8_t luid[8], uint64_t * bytes);

// the physical memory the system can still hand out without paging (Windows: free and standby pages, Linux:
// MemAvailable), 0 when unknown
uint64_t ggml_vk_fork_host_avail_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (uint64_t) st.ullAvailPhys : 0;
#elif defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            break;
        }
    }
    fclose(f);
    return (uint64_t) kb * 1024;
#else
    return 0;
#endif
}

// the local (dedicated) video memory all processes together hold on the adapter with this LUID (VkPhysicalDeviceIDProperties
// deviceLUID: LowPart, then HighPart), by the performance counter Task Manager shows: \GPU Adapter Memory(*)\Dedicated
// Usage, one instance per physical adapter, named luid_0x<HighPart>_0x<LowPart>_phys_<n>. false when the system has no
// such counter or no instance of the adapter (not Windows, before Windows 10 1709)
bool ggml_vk_fork_adapter_dedicated_usage(const uint8_t luid[8], uint64_t * bytes) {
#if defined(_WIN32)
    static std::mutex   mtx;
    static PDH_HQUERY   query   = nullptr;
    static PDH_HCOUNTER counter = nullptr;
    static bool         failed  = false;

    std::lock_guard<std::mutex> lock(mtx);
    if (failed) {
        return false;
    }
    if (query == nullptr) {
        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
            query  = nullptr;
            failed = true;
            return false;
        }
        if (PdhAddEnglishCounterW(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &counter) != ERROR_SUCCESS) {
            PdhCloseQuery(query);
            query  = nullptr;
            failed = true;
            return false;
        }
    }
    // a raw count: one sample gives its value
    if (PdhCollectQueryData(query) != ERROR_SUCCESS) {
        return false;
    }
    DWORD size  = 0;
    DWORD count = 0;
    if ((DWORD) PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &count, nullptr) != (DWORD) PDH_MORE_DATA || size == 0) {
        return false;
    }
    std::vector<uint8_t> buf(size);
    auto * items = (PDH_FMT_COUNTERVALUE_ITEM_W *) buf.data();
    if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &size, &count, items) != ERROR_SUCCESS) {
        return false;
    }

    uint32_t lo = 0;
    uint32_t hi = 0;
    memcpy(&lo, luid, 4);
    memcpy(&hi, luid + 4, 4);
    wchar_t prefix[64];
    swprintf(prefix, 64, L"luid_0x%08X_0x%08X_", (unsigned) hi, (unsigned) lo);
    const size_t n_prefix = wcslen(prefix);

    uint64_t sum   = 0;
    bool     found = false;
    for (DWORD i = 0; i < count; ++i) {
        // a valid value comes with either status: NEW_DATA when it differs from the sample before
        const DWORD status = items[i].FmtValue.CStatus;
        if (items[i].szName != nullptr && _wcsnicmp(items[i].szName, prefix, n_prefix) == 0 &&
                (status == PDH_CSTATUS_VALID_DATA || status == PDH_CSTATUS_NEW_DATA) && items[i].FmtValue.largeValue >= 0) {
            sum  += (uint64_t) items[i].FmtValue.largeValue;
            found = true;
        }
    }
    if (found) {
        *bytes = sum;
    }
    return found;
#else
    (void) luid;
    (void) bytes;
    return false;
#endif
}
