// fork: elastic buffers, included by ggml-cuda.cu after the buffer interface. The elastic KV cache of llama
// (llama_kv_cache::elastic_init_fork, LLAMA_KV_ELASTIC) places its tensors in a buffer that reserves the virtual range
// of the whole cache and maps device memory only to the cells in use, per VMM granule (ggml_backend_buffer_commit_fork,
// ggml_backend_buffer_decommit_fork). The memory the cache does not map stays free for others - the moe-cache, whose
// slots are sized from the free memory of the device. Each granule is created and mapped on its own, so that it can be
// unmapped on its own again. The buffer keeps the CUDA buffer interface: the backend sees an ordinary device buffer.

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

struct ggml_cuda_elastic_fork {
    CUdeviceptr          base            = 0;
    size_t               size            = 0; // reserved, a multiple of the granularity
    size_t               granularity     = 0;
    int                  physical_device = 0;
    std::vector<uint8_t> mapped;              // per granule
};

static void ggml_cuda_elastic_free_fork(ggml_cuda_elastic_fork * el) {
    for (size_t i = 0; i < el->mapped.size(); ++i) {
        if (el->mapped[i]) {
            CU_CHECK(cuMemUnmap(el->base + i*el->granularity, el->granularity));
        }
    }
    CU_CHECK(cuMemAddressFree(el->base, el->size));
    delete el;
}

static void ggml_cuda_elastic_clear_fork(ggml_cuda_elastic_fork * el, uint8_t value) {
    for (size_t i = 0; i < el->mapped.size(); ++i) {
        if (el->mapped[i]) {
            CUDA_CHECK(cudaMemsetAsync((void *) (el->base + i*el->granularity), value, el->granularity, cudaStreamPerThread));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// a buffer of `size` bytes with nothing mapped yet, nullptr where the device has no VMM or the range cannot be reserved
static ggml_backend_buffer_t ggml_backend_cuda_buft_alloc_elastic_fork(ggml_backend_buffer_type_t buft, size_t size) {
    if (!ggml_backend_buft_is_cuda(buft)) {
        return nullptr;
    }
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    const int device = buft_ctx->device;
    if (!ggml_cuda_info().devices[device].vmm || ggml_cuda_info().devices[device].vmm_granularity == 0) {
        return nullptr;
    }

    ggml_cuda_set_device(device);

    ggml_cuda_elastic_fork * el = new ggml_cuda_elastic_fork();
    el->granularity     = ggml_cuda_info().devices[device].vmm_granularity;
    el->physical_device = ggml_cuda_get_physical_device(device);
    el->size            = ((std::max<size_t>(size, 1) + el->granularity - 1)/el->granularity)*el->granularity;

    if (cuMemAddressReserve(&el->base, el->size, 0, 0, 0) != CUDA_SUCCESS) {
        GGML_LOG_ERROR("%s: reserving %.2f MiB of addresses on device %d failed\n", __func__, el->size/1024.0/1024.0, device);
        delete el;
        return nullptr;
    }
    el->mapped.assign(el->size/el->granularity, 0);

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(device, (void *) el->base);
    ctx->elastic = el;

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

// maps device memory, zeroed, to the granules of [offset, offset + size) that have none; false when the device has no
// memory left for them (the granules mapped before stay mapped)
static bool ggml_backend_cuda_buffer_commit_fork(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    if (!ggml_backend_buffer_is_cuda(buffer)) {
        return false;
    }
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    ggml_cuda_elastic_fork * el = ctx->elastic;
    if (el == nullptr || size == 0) {
        return true;
    }

    ggml_cuda_set_device(ctx->device);

    const size_t g  = el->granularity;
    const size_t i0 = offset/g;
    const size_t i1 = std::min(el->mapped.size(), (offset + size + g - 1)/g);

    bool ok = true;
    for (size_t i = i0; i < i1 && ok; ++i) {
        if (el->mapped[i]) {
            continue;
        }

        CUmemAllocationProp prop = {};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = el->physical_device;

        CUmemGenericAllocationHandle handle;
        if (cuMemCreate(&handle, g, &prop, 0) != CUDA_SUCCESS) {
            ok = false;
            break;
        }

        const CUdeviceptr ptr = el->base + i*g;
        if (cuMemMap(ptr, g, 0, handle, 0) != CUDA_SUCCESS) {
            CU_CHECK(cuMemRelease(handle));
            ok = false;
            break;
        }
        // the mapping keeps the memory, the handle is not needed any more
        CU_CHECK(cuMemRelease(handle));

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = el->physical_device;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(ptr, g, &access, 1));

        // padding cells are read by the attention: no NaN in them, as the clear of an ordinary buffer
        CUDA_CHECK(cudaMemsetAsync((void *) ptr, 0, g, cudaStreamPerThread));

        el->mapped[i] = 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    return ok;
}

// unmaps the granules that lie whole inside [offset, offset + size)
static void ggml_backend_cuda_buffer_decommit_fork(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    if (!ggml_backend_buffer_is_cuda(buffer)) {
        return;
    }
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    ggml_cuda_elastic_fork * el = ctx->elastic;
    if (el == nullptr || size == 0) {
        return;
    }

    ggml_cuda_set_device(ctx->device);

    // no kernel may still read the range
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t g  = el->granularity;
    const size_t i0 = (offset + g - 1)/g;
    const size_t i1 = std::min(el->mapped.size(), (offset + size)/g);

    for (size_t i = i0; i < i1; ++i) {
        if (el->mapped[i]) {
            CU_CHECK(cuMemUnmap(el->base + i*g, g));
            el->mapped[i] = 0;
        }
    }
}

#else

struct ggml_cuda_elastic_fork {};

static void ggml_cuda_elastic_free_fork(ggml_cuda_elastic_fork * el) {
    GGML_UNUSED(el);
}

static void ggml_cuda_elastic_clear_fork(ggml_cuda_elastic_fork * el, uint8_t value) {
    GGML_UNUSED(el);
    GGML_UNUSED(value);
}

#endif // defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
