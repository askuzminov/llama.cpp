// fork: no device allocation past the free memory of the device (12.11), included by ggml-cuda.cu before
// ggml_cuda_device_malloc. The driver of Windows backs an allocation that does not fit the free memory with system
// memory when its sysmem fallback policy allows it, which is its default ("CUDA - Sysmem Fallback Policy", drivers
// since 536.40): silently and about 25 times slower (the 3090 on 02.11 with 30 expert layers on the card; the guard of
// 01 on 12.11: 32356 MiB of weights on a card with 23332 MiB free loaded). The policy of the driver stays as it is (the
// user, 12.11), so the allocations that would need it fail here, with the numbers: ggml_cuda_device_malloc (the
// buffers, the legacy pool, which flushes its cached buffers and tries again), the VMM pool (it cannot fail: an abort
// that names LLAMA_MOE_CACHE_MARGIN_MIB) and the elastic buffers of the KV cache (fork/elastic-buffer-fork.inc, which
// keeps GGML_CUDA_ELASTIC_MARGIN_MIB (512) free next to them: a refused growth limits the cells, llama gives up slots of
// the moe-cache for them and tries again).
// GGML_CUDA_ALLOW_SYSMEM_FALLBACK=1 leaves it to the driver again; GGML_CUDA_DEVICE_MARGIN_MIB (0) keeps that much more
// free (the guard of 01 sets it past the card, as GGML_VK_DEVICE_MARGIN_MIB on Vulkan). Windows only: elsewhere an
// allocation past the free memory fails in the driver anyway.

#if defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    define GGML_CUDA_ROOM_FORK 1
#else
#    define GGML_CUDA_ROOM_FORK 0
#endif

// whether size bytes fit the free memory of device (cudaMemGetInfo) next to GGML_CUDA_DEVICE_MARGIN_MIB; what names the
// caller in the error line of a refusal, nullptr for a caller that handles the refusal on its own
static bool ggml_cuda_device_has_room_fork(int device, size_t size, const char * what) {
#if GGML_CUDA_ROOM_FORK
    static const bool   allow  = ggml_cuda_env_flag("GGML_CUDA_ALLOW_SYSMEM_FALLBACK");
    static const size_t margin = (size_t) std::max(0, ggml_cuda_env_int("GGML_CUDA_DEVICE_MARGIN_MIB", 0))*1024*1024;
    if (size == 0 || allow || ggml_cuda_unified_memory_enabled()) {
        return true;
    }
    ggml_cuda_set_device(device);
    size_t free  = 0;
    size_t total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        (void) cudaGetLastError();
        return true; // no numbers: the driver decides, as before
    }
    if (free >= margin && size <= free - margin) {
        return true;
    }
    if (what != nullptr) {
        GGML_LOG_ERROR("%s: %.2f MiB on device %d do not fit its free memory (%.0f MiB free of %.0f MiB, %.0f MiB kept "
                       "free): refused, the driver would put them in system memory (GGML_CUDA_ALLOW_SYSMEM_FALLBACK=1 "
                       "lets it)\n", what, size/1024.0/1024.0, device, free/1024.0/1024.0, total/1024.0/1024.0,
                       margin/1024.0/1024.0);
    }
    return false;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(size);
    GGML_UNUSED(what);
    return true;
#endif
}
