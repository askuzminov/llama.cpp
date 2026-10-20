#include "common-fork.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

//
// common_prompt_checkpoint
//

bool common_prompt_checkpoint::apply(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return false;
    }
    if (data_tgt.empty()) {
        COM_ERR("%s", "no checkpoint data\n");
        return false;
    }

    if (base_pos < 0) {
        // First checkpoint in chain: full restore
        size_t n = llama_state_seq_set_data_ext(
            ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
        if (n != data_tgt.size()) {
            COM_ERR("checkpoint size mismatch: expected %zu, got %zu\n", data_tgt.size(), n);
            return false;
        }
    } else {
        // Subsequent checkpoint: apply delta on top of previous state
        int32_t result = llama_state_seq_apply_delta(
            ctx, data_tgt.data(), data_tgt.size(), seq_id, flags, base_pos);

        if (result != 0) {
            COM_ERR("failed to apply delta (result = %d, base_pos = %d)\n", result, base_pos);
            return false;
        }
    }

    return true;
}

bool common_prompt_checkpoint::apply_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr || data_dft.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    if (n != data_dft.size()) {
        COM_ERR("draft checkpoint size mismatch: expected %zu, got %zu\n", data_dft.size(), n);
        return false;
    }

    return true;
}

//
// Host memory introspection
//

size_t common_host_mem_total() {
#if defined(_WIN32)
    MEMORYSTATUSEX st; st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (size_t) st.ullTotalPhys;
    }
    return 0;
#elif defined(__APPLE__) && defined(__MACH__)
    int64_t mem = 0;
    size_t  len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0 && mem > 0) {
        return (size_t) mem;
    }
    return 0;
#elif defined(__linux__)
    const long pages     = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return (size_t) pages * (size_t) page_size;
    }
    return 0;
#else
    return 0;
#endif
}

size_t common_host_mem_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX st; st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return (size_t) st.ullAvailPhys;
    }
    return 0;
#elif defined(__APPLE__) && defined(__MACH__)
    const mach_port_t host = mach_host_self();

    size_t res = 0;

    vm_size_t page_size = 0;

    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_page_size(host, &page_size) == KERN_SUCCESS && page_size > 0 &&
        host_statistics64(host, HOST_VM_INFO64, (host_info64_t) &vm, &count) == KERN_SUCCESS) {
        // free + file-backed + purgeable, which is what the system can hand out without swapping.
        // inactive is not credited: anonymous inactive pages only move to the compressor. speculative
        // pages are in free_count and in external_page_count, so they are subtracted once.
        const uint64_t free_pages = vm.free_count > vm.speculative_count ? (uint64_t) vm.free_count - vm.speculative_count : 0;

        const uint64_t avail_pages = free_pages + vm.external_page_count + vm.purgeable_count;

        res = (size_t) (avail_pages * (uint64_t) page_size);
    }

    mach_port_deallocate(mach_task_self(), host);

    return res;
#elif defined(__linux__)
    // Prefer MemAvailable: the kernel's estimate that accounts for reclaimable page cache.
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("MemAvailable:", 0) == 0) {
                unsigned long long kb = 0;
                if (sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1) {
                    return (size_t) kb * 1024;
                }
            }
        }
    }

    const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size   = sysconf(_SC_PAGESIZE);
    if (avail_pages > 0 && page_size > 0) {
        return (size_t) avail_pages * (size_t) page_size;
    }
    return 0;
#else
    return 0;
#endif
}
