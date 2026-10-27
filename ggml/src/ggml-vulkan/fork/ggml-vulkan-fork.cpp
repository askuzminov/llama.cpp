// device setup, tile overrides, the shared buffer type and graph timing of the fork (see ggml-vulkan-fork.h)

#include "ggml-vulkan-fork.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>

// runs a probe shader with one storage buffer of `words` 32-bit words, zeroed, on n_wg workgroups and copies the
// buffer to `out`. spec: the specialization constants 0.., subgroup_size: a required subgroup size or 0. false when
// the driver refuses the shader or it does not complete. plain Vulkan calls: it runs before the device has its pipelines
static bool ggml_vk_fork_run_probe(vk_device & device, const char * what, const void * spv, size_t spv_len, const std::vector<uint32_t> & spec,
                                   uint32_t n_wg, uint32_t words, uint32_t subgroup_size, std::vector<uint32_t> & out) {
    vk::Device d = device->device;
    const vk::DeviceSize size = (vk::DeviceSize) words * sizeof(uint32_t);

    vk::ShaderModule shader;
    vk::DescriptorSetLayout dsl;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DescriptorPool pool;
    vk::CommandPool cmd_pool;
    vk::Fence fence;
    uint32_t * data = nullptr;
    bool submitted = false;
    bool ok = false;

    try {
        shader = d.createShaderModule({ {}, spv_len, reinterpret_cast<const uint32_t *>(spv) });
        const vk::DescriptorSetLayoutBinding binding { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute };
        dsl = d.createDescriptorSetLayout({ {}, binding });
        layout = d.createPipelineLayout({ {}, dsl });
        std::vector<vk::SpecializationMapEntry> entries;
        for (uint32_t i = 0; i < (uint32_t) spec.size(); i++) {
            entries.push_back({ i, i * (uint32_t) sizeof(uint32_t), sizeof(uint32_t) });
        }
        const vk::SpecializationInfo spec_info { (uint32_t) entries.size(), entries.data(), spec.size() * sizeof(uint32_t), spec.data() };
        vk::PipelineShaderStageCreateInfo stage { {}, vk::ShaderStageFlagBits::eCompute, shader, "main", spec.empty() ? nullptr : &spec_info };
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_info;
        subgroup_info.requiredSubgroupSize = subgroup_size;
        if (subgroup_size != 0) {
            if (!device->subgroup_size_control || subgroup_size < device->subgroup_min_size || subgroup_size > device->subgroup_max_size) {
                throw std::runtime_error("the subgroup size is not available");
            }
            stage.setPNext(&subgroup_info);
            stage.flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
        }
        pipeline = d.createComputePipeline(nullptr, vk::ComputePipelineCreateInfo { {}, stage, layout }).value;

        buffer = d.createBuffer({ {}, size, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive });
        const vk::MemoryRequirements req = d.getBufferMemoryRequirements(buffer);
        const vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();
        const vk::MemoryPropertyFlags host = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        uint32_t type_idx = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount && type_idx == UINT32_MAX; i++) {
            if ((req.memoryTypeBits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & host) == host) {
                type_idx = i;
            }
        }
        if (type_idx == UINT32_MAX) {
            throw std::runtime_error("no host visible memory type");
        }
        memory = d.allocateMemory({ req.size, type_idx });
        d.bindBufferMemory(buffer, memory, 0);
        data = static_cast<uint32_t *>(d.mapMemory(memory, 0, size));
        memset(data, 0, size);

        const vk::DescriptorPoolSize pool_size { vk::DescriptorType::eStorageBuffer, 1 };
        pool = d.createDescriptorPool({ {}, 1, pool_size });
        const vk::DescriptorSet set = d.allocateDescriptorSets({ pool, dsl })[0];
        const vk::DescriptorBufferInfo buf_info { buffer, 0, size };
        d.updateDescriptorSets(vk::WriteDescriptorSet { set, 0, 0, vk::DescriptorType::eStorageBuffer, {}, buf_info }, {});

        cmd_pool = d.createCommandPool({ vk::CommandPoolCreateFlagBits::eTransient, device->compute_queue->queue_family_index });
        const vk::CommandBuffer cmd = d.allocateCommandBuffers({ cmd_pool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, set, {});
        cmd.dispatch(n_wg, 1, 1);
        // the shader writes become visible to the host read. the barrier type stays unnamed: windows.h defines
        // MemoryBarrier as a macro
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {},
                            { { { vk::AccessFlagBits::eShaderWrite }, { vk::AccessFlagBits::eHostRead } } }, {}, {});
        cmd.end();

        fence = d.createFence({});
        device->compute_queue->handle->submit(vk::SubmitInfo { {}, {}, cmd }, fence);
        submitted = true;
        if (d.waitForFences(fence, true, UINT64_MAX) != vk::Result::eSuccess) {
            throw std::runtime_error("probe did not complete");
        }
        submitted = false;

        out.assign(data, data + words);
        ok = true;
    } catch (const std::exception & e) {
        GGML_LOG_WARN("ggml_vulkan: %s probe failed: %s\n", what, e.what());
        if (submitted) {
            d.waitIdle();
        }
        ok = false;
    }

    if (data) d.unmapMemory(memory);
    if (fence) d.destroyFence(fence);
    if (cmd_pool) d.destroyCommandPool(cmd_pool);
    if (pool) d.destroyDescriptorPool(pool);
    if (buffer) d.destroyBuffer(buffer);
    if (memory) d.freeMemory(memory);
    if (pipeline) d.destroyPipeline(pipeline);
    if (layout) d.destroyPipelineLayout(layout);
    if (dsl) d.destroyDescriptorSetLayout(dsl);
    if (shader) d.destroyShaderModule(shader);
    return ok;
}

// runs shmem_probe.comp with `bytes` of shared memory per workgroup. true when the driver takes the shader and
// every workgroup reads back its own data
static bool ggml_vk_shmem_probe(vk_device & device, uint32_t bytes) {
    const uint32_t words = bytes / (uint32_t) sizeof(uint32_t);
    const uint32_t n_wg = 512;
    std::vector<uint32_t> errors;
    if (!ggml_vk_fork_run_probe(device, "shared memory", shmem_probe_data, shmem_probe_len, { words }, n_wg, n_wg, 0, errors)) {
        return false;
    }
    uint64_t bad = 0;
    for (uint32_t i = 0; i < n_wg; i++) {
        bad += errors[i];
    }
    if (bad != 0) {
        GGML_LOG_WARN("ggml_vulkan: shared memory probe of %u bytes: %llu words of %llu read back wrong\n", bytes,
            (unsigned long long) bad, (unsigned long long) 4 * n_wg * words);
    }
    return bad == 0;
}

bool ggml_vk_fork_coopmat_a_probe(vk_device & device) {
#if !(defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT))
    GGML_UNUSED(device);
    return false;
#else
    // coopmat_a_probe.comp: element e of an f16 16x16 A fragment of invocation l is row l % 16, column e (wave32)
    std::vector<uint32_t> d;
    if (!ggml_vk_fork_run_probe(device, "coopmat A layout", coopmat_a_probe_cm1_data, coopmat_a_probe_cm1_len, {}, 1, 257, 32, d)) {
        return false;
    }
    float v[257];
    memcpy(v, d.data(), sizeof(v));
    int bad = 0;
    for (int i = 0; i < 256; i++) {
        bad += v[i] != (float) i;
    }
    if (bad != 0 || v[256] != 16.0f) {
        GGML_LOG_INFO("ggml_vulkan: coopmat A layout probe: %d of 256 elements elsewhere, fragment length %g\n", bad, (double) v[256]);
        return false;
    }
    return true;
#endif
}

void ggml_vk_fork_mmid_f16reg_init(vk_device & device) {
    // the experts' prefill matmuls on mul_mmid_f16reg.comp (f16 coopmat, the codes decoded into the A fragments) where
    // coopmat_a_probe.comp finds the A layout it takes. the default on RDNA3 since 11.10: on the 395 a 4096-token prompt
    // 4367 -> 4060 ms of gpu time at depth 0 and 5182 -> 4908 at 122880 (fused gate/up 877 -> 601), 02 PP +6.6 percent
    // at 0, KLD 0.015735 against 0.016057 (f16 activations instead of the 8-bit ones of the int8 path).
    // GGML_VK_MMID_F16REG=0: the int8 path, =1: on any device that passes the probe, =emulate: the same algorithm with
    // subgroup shuffles on any device with 32-invocation subgroups (tests)
    device->mmid_f16reg_mode = 0;
    const char * env = getenv("GGML_VK_MMID_F16REG");
    if (env != nullptr && strcmp(env, "0") == 0) {
        return;
    }
    if (env == nullptr) {
        if (!(device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA3)) {
            return;
        }
        env = "1";
    }
    const bool sg32 = device->subgroup_size_control ? (device->subgroup_min_size <= 32 && 32 <= device->subgroup_max_size)
                                                     : device->subgroup_size == 32;
    if (!device->fp16 || !sg32) {
        GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMID_F16REG needs fp16 and 32-invocation subgroups\n");
        return;
    }
    if (strcmp(env, "emulate") == 0) {
        device->mmid_f16reg_mode = 2;
        GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID f16reg emulated with subgroup shuffles\n");
        return;
    }
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (!device->coopmat_support || !device->subgroup_size_control) {
        GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMID_F16REG needs KHR coopmat and the subgroup size control\n");
        return;
    }
    if (!ggml_vk_fork_coopmat_a_probe(device)) {
        GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMID_F16REG off: the coopmat A layout is not the one the shader takes\n");
        if (getenv("GGML_SCHED_LOG_REALLOC")) {
            fprintf(stderr, "ggml_vulkan: GGML_VK_MMID_F16REG off: the coopmat A layout is not the one the shader takes\n");
        }
        return;
    }
    device->mmid_f16reg_mode = 1;
    GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID f16reg on (coopmat A layout probe passed)\n");
    if (getenv("GGML_SCHED_LOG_REALLOC")) {
        fprintf(stderr, "ggml_vulkan: MUL_MAT_ID f16reg on (coopmat A layout probe passed)\n");
    }
#else
    GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMID_F16REG needs a build with coopmat shaders\n");
#endif
}

void ggml_vk_fork_fa_union_init(vk_device & device) {
    // the sparse attention of a prompt with a tile of neighbouring tokens over the union of their selections
    // (flash_attn_union.comp, see ggml_vk_fork_fa_union). GGML_VK_FA_UNION: unset or 0 = off (until measured), 1 = on
    // with coopmat, emulate = the same steps without coopmat on any device (tests)
    device->fa_union_mode = 0;
    const char * env = getenv("GGML_VK_FA_UNION");
    if (env == nullptr || strcmp(env, "0") == 0) {
        return;
    }
    const uint32_t sg = device->subgroup_size;
    if (!device->fp16 || !device->subgroup_require_full_support || (sg != 16 && sg != 32 && sg != 64)) {
        GGML_LOG_WARN("ggml_vulkan: GGML_VK_FA_UNION needs fp16 and full subgroups of 16, 32 or 64\n");
        return;
    }
    if (strcmp(env, "emulate") == 0) {
        device->fa_union_mode = 2;
        GGML_LOG_INFO("ggml_vulkan: sparse prompt attention over the union of neighbouring tokens, emulated\n");
        return;
    }
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (!device->coopmat_support || !device->coopmat_support_16x16x16_f32acc) {
        GGML_LOG_WARN("ggml_vulkan: GGML_VK_FA_UNION needs KHR coopmat 16x16x16 with f32 accumulators\n");
        return;
    }
    device->fa_union_mode = 1;
    GGML_LOG_INFO("ggml_vulkan: sparse prompt attention over the union of neighbouring tokens\n");
#else
    GGML_LOG_WARN("ggml_vulkan: GGML_VK_FA_UNION needs a build with coopmat shaders\n");
#endif
}

void ggml_vk_fork_shmem_limit(vk_device & device) {
    // the shared memory in bytes that the shaders may declare. above the limit the driver reports, a shader is
    // outside the Vulkan spec: the driver may refuse it, run it, compute garbage or hang the device. the AMD
    // driver on Windows reports 32 KB where RDNA has 64 KB per workgroup; on the RDNA3.5 iGPU of Strix Halo
    // every test passed with 64 KB (matmul, matmul_id, flash attention, solve_tri; 02.10.2026), so the RDNA3
    // iGPUs on that driver ask for 64 KB. GGML_VK_SHMEM_LIMIT (bytes) overrides, 32768 keeps the reported
    // limit. this runs once the compute queue exists: a raised limit is applied only after shmem_probe.comp has
    // passed with it
    uint32_t shmem_limit = device->properties.limits.maxComputeSharedMemorySize;
    const char * shmem_why = nullptr;
    if (device->driver_id == vk::DriverId::eAmdProprietary && device->architecture == AMD_RDNA3 &&
        device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && shmem_limit < 65536) {
        shmem_limit = 65536;
        shmem_why = "RDNA3 iGPU on the AMD driver";
    }
    if (const char * shmem_env = getenv("GGML_VK_SHMEM_LIMIT")) {
        if (atoi(shmem_env) > 0) {
            shmem_limit = (uint32_t) atoi(shmem_env);
            shmem_why = "GGML_VK_SHMEM_LIMIT";
        }
    }

    // the shared memory limit chosen above: a lower one is taken as it is, a raised one only after the probe
    const uint32_t reported = device->properties.limits.maxComputeSharedMemorySize;
    if (shmem_limit < reported) {
        GGML_LOG_WARN("ggml_vulkan: shared memory limit %u -> %u (%s)\n", reported, shmem_limit, shmem_why);
        device->properties.limits.maxComputeSharedMemorySize = shmem_limit;
    } else if (shmem_limit > reported) {
        if (ggml_vk_shmem_probe(device, shmem_limit)) {
            GGML_LOG_WARN("ggml_vulkan: shared memory limit %u -> %u (%s), outside the Vulkan spec, the probe passed\n",
                reported, shmem_limit, shmem_why);
            device->properties.limits.maxComputeSharedMemorySize = shmem_limit;
        } else {
            GGML_LOG_WARN("ggml_vulkan: shared memory limit stays %u: %u bytes (%s) failed the probe\n",
                reported, shmem_limit, shmem_why);
        }
    }
    // GGML_VK_SHMEM_PROBE=1 (diagnostics): run the probe on the limit in use as well
    if (getenv("GGML_VK_SHMEM_PROBE") != nullptr) {
        const uint32_t limit = device->properties.limits.maxComputeSharedMemorySize;
        GGML_LOG_INFO("ggml_vulkan: shared memory probe of %u bytes: %s\n", limit,
            ggml_vk_shmem_probe(device, limit) ? "passed" : "failed");
    }
}

void ggml_vk_fork_log_sparse(const vk_device & device) {
    const vk::PhysicalDeviceFeatures features = device->physical_device.getFeatures();
    const std::vector<vk::QueueFamilyProperties> families = device->physical_device.getQueueFamilyProperties();
    size_t n_sparse = 0;
    for (const auto & f : families) {
        n_sparse += (f.queueFlags & vk::QueueFlagBits::eSparseBinding) ? 1 : 0;
    }
    GGML_LOG_INFO("ggml_vulkan: sparse buffers: binding %d, residency %d, %zu of %zu queue families, max allocations %u\n",
        (int) features.sparseBinding, (int) features.sparseResidencyBuffer, n_sparse, families.size(),
        device->properties.limits.maxMemoryAllocationCount);
}

void ggml_vk_fork_log_limits(const vk_device & device) {
    // GGML_SCHED_LOG_REALLOC (ggml-backend.cpp): the limits behind the tensors supports_op sends to another backend
    if (getenv("GGML_SCHED_LOG_REALLOC")) {
        fprintf(stderr, "ggml_vulkan: max buffer size %llu, max storage buffer range %u, max allocation %llu, 64-bit indexing %d\n",
                (unsigned long long) device->max_buffer_size, device->properties.limits.maxStorageBufferRange,
                (unsigned long long) device->max_memory_allocation_size, (int) device->shader_64b_indexing);
    }
}

size_t ggml_vk_fork_host_buffer_pad(const vk_device & device, size_t size) {
    // upstream adds 32 bytes to every pinned host buffer, as the CPU buffer type once did (it no longer pads). a buffer
    // of exactly the device limit then goes over it and falls back to pageable memory with "Failed to allocate pinned
    // memory": on the 395 (AMD driver, 2 GiB per buffer) the f16 KQ mask of a 4096-token ubatch planned at 262144
    // cells is 2 GiB and takes a chunk of its own in the host compute buffer, of the target and of the MTP draft
    // context (09.10). the mapping is aligned to minMemoryMapAlignment and the chunk holds its tensors without slack.
    // GGML_VK_HOST_PAD_UPSTREAM=1: always pad
    static const bool upstream = [] {
        const char * env = getenv("GGML_VK_HOST_PAD_UPSTREAM");
        return env != nullptr && atoi(env) != 0;
    }();
    const size_t pad   = 32;
    const size_t limit = (size_t) std::min<uint64_t>(device->max_buffer_size, device->max_memory_allocation_size);
    if (upstream || size > limit || size + pad <= limit) {
        return pad;
    }
    if (getenv("GGML_SCHED_LOG_REALLOC")) {
        fprintf(stderr, "ggml_vulkan: pinned host buffer of %zu bytes without the %zu-byte pad (device limit %zu)\n", size, pad, limit);
    }
    return 0;
}

bool ggml_vk_fork_int_coopmat(vk_device & device, int & mode) {
    // GGML_VK_INT_COOPMAT: 0 int8 coopmat off, 1 on for MUL_MAT and MUL_MAT_ID, 2 for MUL_MAT only, 3 for
    // MUL_MAT_ID only, on any driver (see below); unset keeps the default
    // mul_mmq_cm1 hardcodes the RDNA3 and RDNA4 accumulator layouts. RADV gives both; the AMD driver
    // gives the RDNA3 one (Strix Halo, test-backend-ops and KLD, 02.10.2026). #29392 (wrong results
    // with the AMD driver on Windows) came from an old driver without shader_float8, so RDNA4 got the
    // RDNA3 layout; driver 26.10.44 fixed it, RDNA4 on the AMD driver is not verified here
    const char * int_coopmat_env = getenv("GGML_VK_INT_COOPMAT");
    const int int_coopmat_mode = int_coopmat_env ? atoi(int_coopmat_env) : -1;
    const bool int_coopmat_default = device->driver_id == vk::DriverId::eMesaRadv ||
                                     (device->driver_id == vk::DriverId::eAmdProprietary && device->architecture == AMD_RDNA3);
    const bool int_coopmat_on = int_coopmat_mode < 0 ? int_coopmat_default : int_coopmat_mode >= 1 && int_coopmat_mode <= 3;
    device->coopmat_int_ops = int_coopmat_mode == 2 ? 1u : int_coopmat_mode == 3 ? 2u : 3u;
    mode = int_coopmat_mode;
    return int_coopmat_on;
}

void ggml_vk_fork_log_int_coopmat(const vk_device & device, int mode, uint32_t m, uint32_t n, uint32_t k) {
    if (device->driver_id != vk::DriverId::eMesaRadv || mode > 1) {
        GGML_LOG_INFO("ggml_vulkan: int8 coopmat %ux%ux%u on for %s, architecture %s, subgroup %u-%u\n",
            m, n, k,
            device->coopmat_int_ops == 1 ? "MUL_MAT" : device->coopmat_int_ops == 2 ? "MUL_MAT_ID" : "MUL_MAT and MUL_MAT_ID",
            device->architecture == AMD_RDNA3 ? "RDNA3" : device->architecture == AMD_RDNA4 ? "RDNA4" : "other, no int8 pipelines",
            device->subgroup_min_size, device->subgroup_max_size);
    }
}

void ggml_vk_fork_f16acc(vk_device & device) {
    // f16 accumulators of the coopmat matmuls, off by default on the AMD driver. Strix Halo, qwen4exp, KLD against
    // f32 accumulators: 0.019, twice what a mere reordering of the sums gives (0.010), for 1% of the prefill.
    // GGML_VK_F16ACC=0 or 1 overrides
    const char * f16acc_env = getenv("GGML_VK_F16ACC");
    if (f16acc_env ? atoi(f16acc_env) == 0 : device->driver_id == vk::DriverId::eAmdProprietary) {
        device->coopmat_acc_f16_support = false;
    }
}

void ggml_vk_fork_int_large_tile(const vk_device & device, bool & matmul, bool & matmul_id) {
    // GGML_VK_INT_LARGE_TILE (experiment): the large int8 coopmat tiles even where the large f16 tile is off, as on
    // the AMD driver, where the f16 one spills (28.09). 1 for matmul and matmul_id, 2 for matmul only: on the 395
    // (06.10) the 64x128 tile took 4 percent off the dense matmuls but slowed the experts on a real routing. the
    // shared memory check of ggml_vk_load_shaders still applies
    // the default is 2 on the AMD driver on RDNA3 (with the 64x128 tile, see GGML_VK_MMQ_INT_TILE), 0 elsewhere
    const int int_large_tile_mode = getenv("GGML_VK_INT_LARGE_TILE") != nullptr ? atoi(getenv("GGML_VK_INT_LARGE_TILE")) :
        (device->driver_id == vk::DriverId::eAmdProprietary && device->architecture == AMD_RDNA3 ? 2 : 0);
    const bool int_large_tile    = int_large_tile_mode != 0;
    const bool int_large_tile_id = int_large_tile_mode == 1;
    if (int_large_tile) {
        GGML_LOG_INFO("ggml_vulkan: large int8 coopmat tiles on for matmul%s (GGML_VK_INT_LARGE_TILE)\n",
            int_large_tile_id ? " and matmul_id" : "");
    }
    matmul    = int_large_tile;
    matmul_id = int_large_tile_id;
}

void ggml_vk_fork_device_init(vk_device & device, size_t idx) {
    device->buffer_type_shared = {
        /* .iface    = */ ggml_backend_vk_buffer_type_interface,
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
        /* .context  = */ new ggml_backend_vk_buffer_type_context{ device->name + "_Shared", device, true },
    };
    device->extra_bufts[0] = &device->buffer_type_shared;

    // a matmul whose A and B do not fit the last level cache reads them from memory once per tile. AMD APUs
    // have a 32 MB MALL: keep A and the part of B that is in use to 24 MB. GGML_VK_MM_CHUNK_MB=0 turns this off
    const char * mm_chunk_mb = getenv("GGML_VK_MM_CHUNK_MB");
    if (mm_chunk_mb) {
        device->mm_chunk_bytes = (uint64_t) (std::max(0.0, atof(mm_chunk_mb)) * 1024.0 * 1024.0);
    } else {
        device->mm_chunk_bytes = device->vendor_id == VK_VENDOR_ID_AMD && device->uma ? 24ull*1024*1024 : 0;
    }
}

void ggml_vk_fork_mmq_tile(vk_device & device, const vk_pipeline & requested, std::vector<uint32_t> & m_warptile_mmq,
                           std::vector<uint32_t> & m_warptile_mmqid, std::array<uint32_t, 3> & m_mmq_wg_denoms) {
    // GGML_VK_MMQ_TILE: the medium tile of the quant matmuls and matmul_id, 11 comma separated warptile values:
    // BLOCK_SIZE,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,WARP. for tile sweeps: on some drivers the medium tile is the
    // largest one left, so it runs every large matmul. matmul_id takes the same tile, as coopmat1 does anyway
    const char * mmq_tile = getenv("GGML_VK_MMQ_TILE");
    if (mmq_tile && mmq_tile[0]) {
        std::vector<uint32_t> wt;
        std::stringstream ss(mmq_tile);
        for (std::string v; std::getline(ss, v, ',');) {
            wt.push_back((uint32_t) std::stoul(v));
        }
        // one load pass of A or B covers BLOCK_SIZE*8/BK rows (8 values per thread for q4_0, q5_1, iq4_xs and
        // f16 B) and the load loops have no row guard, so a pass must fit in BM and BN (512 threads on a 64 row
        // side fail the tests)
        if (wt.size() == 11 && wt[0] > 0 && wt[10] > 0 && wt[0] % wt[10] == 0 && wt[4] > 0 && wt[5] > 0 &&
            wt[1] % wt[4] == 0 && wt[2] % wt[5] == 0 && (wt[1] / wt[4]) * (wt[2] / wt[5]) == wt[0] / wt[10] &&
            wt[3] == 32 && (wt[1] & (wt[1] - 1)) == 0 && (wt[2] & (wt[2] - 1)) == 0 &&
            wt[0] * 8 / wt[3] <= wt[1] && wt[0] * 8 / wt[3] <= wt[2]) {
            // shared memory as mul_mm.comp declares it: the A and B stages, the row ids of matmul_id, the coopmat
            // stage of TM*TN accumulators per wave (f32 at worst) and the ballots
            const uint32_t warps = wt[0] / wt[10];
            const uint32_t shmem = (wt[1] + wt[2]) * (wt[3] + (device->coopmat_support ? 8 : 1)) * sizeof(ggml_fp16_t) +
                                   wt[2] * 2 * sizeof(uint16_t) +
                                   (device->coopmat_support ? wt[7] * wt[8] * warps * sizeof(float) : 0) + warps * 4 * sizeof(uint32_t);
            // pipelines compiled on demand run this function again, log only at device init
            if (shmem <= device->properties.limits.maxComputeSharedMemorySize) {
                m_warptile_mmq   = wt;
                m_warptile_mmqid = wt;
                m_mmq_wg_denoms  = { wt[1], wt[2], 1 };
                if (!requested) {
                    GGML_LOG_INFO("ggml_vulkan: medium quant tile %s, shared memory %u of %u\n", mmq_tile, shmem,
                        device->properties.limits.maxComputeSharedMemorySize);
                }
            } else if (!requested) {
                GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMQ_TILE=%s ignored: needs %u bytes of shared memory, the device has %u\n",
                    mmq_tile, shmem, device->properties.limits.maxComputeSharedMemorySize);
            }
        } else if (!requested) {
            // the loads of the quant blocks assume a K step of 32 (q5_1 and the tail of a K that is not a multiple
            // of the step break with 64)
            // the tile edges are rounded with power of two masks (BM 192 fails the tests)
            GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMQ_TILE=%s ignored: needs 11 values, (BM/WM)*(BN/WN) == BLOCK_SIZE/WARP, BK 32, BM and BN powers of two and >= BLOCK_SIZE*8/BK\n",
                mmq_tile);
        }
    }
}

void ggml_vk_fork_int_tiles(vk_device & device, const vk_pipeline & requested, uint32_t cm1_sg, uint32_t itm, uint32_t itn,
                            uint32_t itk, std::vector<uint32_t> & l_warptile_mmq_cm1_int, vk_fork_int_tiles & tiles) {
    // GGML_VK_MMQ_INT_TILE (experiment): the large int8 coopmat tile as "BM,BN,WM,WN". a wave of cm1_sg lanes
    // computes WM x WN, (BM/WM)*(BN/WN) waves make the workgroup. the loads of mul_mmq_cm1 guard the tile
    // edges, so any multiple of 16 works; the shared memory is checked per type in ggml_vk_load_shaders. it takes
    // effect where the large int8 tile is on (RADV, or GGML_VK_INT_LARGE_TILE=1)
    // the AMD driver on RDNA3 takes 64x128 by default, with GGML_VK_INT_LARGE_TILE=2 (07.10, the 395: the gpu
    // time of a 4096-token prompt -1.4 percent, PP +1.5 percent, the experts keep their tile)
    const bool amd_rdna3_driver = device->driver_id == vk::DriverId::eAmdProprietary && device->architecture == AMD_RDNA3;
    const char * mmq_int_tile = getenv("GGML_VK_MMQ_INT_TILE");
    if (mmq_int_tile == nullptr && amd_rdna3_driver) {
        mmq_int_tile = "64,128,32,32";
    }
    if (mmq_int_tile && mmq_int_tile[0] && device->coopmat_int_support) {
        std::vector<uint32_t> v;
        std::stringstream ss(mmq_int_tile);
        for (std::string part; std::getline(ss, part, ',');) {
            v.push_back((uint32_t) std::stoul(part));
        }
        const bool shape_ok = v.size() == 4 && v[2] > 0 && v[3] > 0 && v[2] % 16 == 0 && v[3] % 16 == 0 &&
                              v[0] % v[2] == 0 && v[1] % v[3] == 0;
        const uint32_t block = shape_ok ? cm1_sg * (v[0] / v[2]) * (v[1] / v[3]) : 0;
        if (shape_ok && block > 0 && block <= device->properties.limits.maxComputeWorkGroupInvocations) {
            l_warptile_mmq_cm1_int = { block, v[0], v[1], 32, v[2], v[3], 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
            tiles.l_mmq_int_wg_denoms = { v[0], v[1], 1 };
            tiles.mmq_int_tile_set = true;
            if (!requested) {
                GGML_LOG_INFO("ggml_vulkan: large int8 tile %s: %u threads, q4_K fits in shared memory: matmul %s, matmul_id %s\n",
                    mmq_int_tile, block,
                    ggml_vk_matmul_cm1_int_shmem_support(device, l_warptile_mmq_cm1_int, false, GGML_TYPE_Q4_K, false) ? "yes" : "no",
                    ggml_vk_matmul_cm1_int_shmem_support(device, l_warptile_mmq_cm1_int, true, GGML_TYPE_Q4_K, false) ? "yes" : "no");
            }
        } else if (!requested) {
            GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMQ_INT_TILE=%s ignored: needs BM,BN,WM,WN with WM, WN multiples of 16 that divide BM, BN and at most %u threads\n",
                mmq_int_tile, device->properties.limits.maxComputeWorkGroupInvocations);
        }
    }

    // the matmul_id tile: the experts see few rows each (about 80 per expert at -ub 4096 with 512 experts, 10 per
    // token), so their best tile is not that of the dense matmuls (06.10: 64x128 sped the dense ones up and slowed
    // the experts). takes effect where the large int8 tile is on for matmul_id (GGML_VK_INT_LARGE_TILE=1)
    // "BM,BN,WM,WN" as GGML_VK_MMQ_INT_TILE into the warptile and workgroup size of a matmul_id tile
    const auto parse_int_tile_id = [&](const char * env, const char * value, bool glu, std::vector<uint32_t> & wt,
                                       std::array<uint32_t, 3> & denoms) -> bool {
        std::vector<uint32_t> v;
        std::stringstream ss(value);
        for (std::string part; std::getline(ss, part, ',');) {
            v.push_back((uint32_t) std::stoul(part));
        }
        const bool shape_ok = v.size() == 4 && v[2] > 0 && v[3] > 0 && v[2] % 16 == 0 && v[3] % 16 == 0 &&
                              v[0] % v[2] == 0 && v[1] % v[3] == 0;
        const uint32_t block = shape_ok ? cm1_sg * (v[0] / v[2]) * (v[1] / v[3]) : 0;
        if (!shape_ok || block == 0 || block > device->properties.limits.maxComputeWorkGroupInvocations) {
            if (!requested) {
                GGML_LOG_WARN("ggml_vulkan: %s=%s ignored: needs BM,BN,WM,WN as GGML_VK_MMQ_INT_TILE\n", env, value);
            }
            return false;
        }
        wt     = { block, v[0], v[1], 32, v[2], v[3], 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        denoms = { v[0], v[1], 1 };
        if (!requested) {
            GGML_LOG_INFO("ggml_vulkan: %s %s: %u threads, q4_K fits in shared memory: %s\n", env, value, block,
                ggml_vk_matmul_cm1_int_shmem_support(device, wt, true, GGML_TYPE_Q4_K, glu) ? "yes" : "no");
        }
        return true;
    };

    tiles.l_warptile_mmq_cm1_int_id = l_warptile_mmq_cm1_int;
    tiles.l_mmq_int_id_wg_denoms    = tiles.l_mmq_int_wg_denoms;
    tiles.mmq_int_tile_id_set       = tiles.mmq_int_tile_set;
    const char * mmq_int_tile_id = getenv("GGML_VK_MMQ_INT_TILE_ID");
    if (mmq_int_tile_id && mmq_int_tile_id[0] && device->coopmat_int_support &&
        parse_int_tile_id("GGML_VK_MMQ_INT_TILE_ID", mmq_int_tile_id, false, tiles.l_warptile_mmq_cm1_int_id, tiles.l_mmq_int_id_wg_denoms)) {
        tiles.mmq_int_tile_id_set = true;
    }

    // the fused gate/up of the experts (m=640 on qwen4exp) on a large tile of its own, the down projection keeps
    // the medium one. 07.10, the 395 at -ub 4096: 32x64 took the gate/up from 933 to 871 ms of gpu time and slowed
    // the down projection (m=2560) from 529 to 574 ms, 64x32, 128x32 and 128x64 slowed the gate/up. 0 turns it off
    const char * mmq_int_tile_glu = getenv("GGML_VK_MMQ_INT_TILE_GLU");
    if (mmq_int_tile_glu == nullptr && amd_rdna3_driver) {
        mmq_int_tile_glu = "32,64,16,32";
    }
    if (mmq_int_tile_glu && mmq_int_tile_glu[0] && strcmp(mmq_int_tile_glu, "0") != 0 && device->coopmat_int_support &&
        parse_int_tile_id("GGML_VK_MMQ_INT_TILE_GLU", mmq_int_tile_glu, true, tiles.l_warptile_mmq_cm1_int_glu, tiles.l_mmq_int_glu_wg_denoms)) {
        tiles.mmq_int_tile_glu_set = true;
    }
}

// the shared buffer type of the device (<name>_Shared). llama.cpp lists it after the default one, so it holds only
// the tensors an override (-ot) or LLAMA_KV_SHARED puts there
static ggml_backend_buffer_type_t * ggml_backend_vk_device_get_extra_bufts(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *) dev->context;
    ggml_vk_instance_init();
    return ggml_vk_get_device(ctx->device)->extra_bufts;
}

// GGML_TIMELINE: the gpu span of each command buffer in the gpu clock, with the host time its recording ended; they go
// to the timeline at the end of the capture, shifted onto the host clock (ggml_timeline_device_spans)
struct vk_timeline_fork {
    static constexpr uint32_t n_slots = 2048;

    vk::Device    dev;
    vk::QueryPool pool;
    double        period_us = 0.0;
    std::string   track;
    uint32_t      next = 0;

    std::vector<uint8_t>  busy;       // a slot begun and not read back
    std::vector<int64_t>  rec_end_us; // the host time at the end of its recording
    std::vector<uint32_t> pending;    // ended, not read back
    std::unordered_map<VkCommandBuffer, uint32_t> open; // begun, not ended
    std::vector<std::array<double, 3>> spans;           // gpu start and end (us of the gpu clock), end of recording
    int64_t n_dropped = 0;                              // begun while the ring was full
};

static std::mutex vk_timeline_mutex;
static std::atomic<bool> vk_timeline_any { false };
static std::unordered_map<const vk_device_struct *, std::unique_ptr<vk_timeline_fork>> vk_timelines;

void ggml_vk_fork_timeline_begin(vk_device & device, vk_context & subctx) {
    // a transfer queue may have no timestamps
    if (!ggml_timeline_on() || subctx->s == nullptr || subctx->p == nullptr || subctx->p->q->transfer_only) {
        return;
    }
    std::lock_guard<std::mutex> lock(vk_timeline_mutex);
    auto & tl = vk_timelines[device.get()];
    if (!tl) {
        tl = std::make_unique<vk_timeline_fork>();
        vk::QueryPoolCreateInfo info;
        info.queryType  = vk::QueryType::eTimestamp;
        info.queryCount = 2 * vk_timeline_fork::n_slots;
        tl->dev       = device->device;
        tl->pool      = device->device.createQueryPool(info);
        tl->period_us = 1e-3 * device->properties.limits.timestampPeriod;
        tl->track     = "gpu " + device->name;
        tl->busy.assign(vk_timeline_fork::n_slots, 0);
        tl->rec_end_us.assign(vk_timeline_fork::n_slots, 0);
        device->device.resetQueryPool(tl->pool, 0, 2 * vk_timeline_fork::n_slots);
        vk_timeline_any = true;
    }
    // a full ring waits for the next sync: the span is left out
    if (tl->busy[tl->next]) {
        tl->n_dropped++;
        return;
    }
    const uint32_t slot = tl->next;
    tl->next = (tl->next + 1) % vk_timeline_fork::n_slots;
    tl->busy[slot] = 1;
    vk::CommandBuffer buf = subctx->s->buffer->buf;
    buf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, tl->pool, 2 * slot);
    tl->open[(VkCommandBuffer) buf] = slot;
}

void ggml_vk_fork_timeline_end(vk_context & ctx) {
    if (!vk_timeline_any || ctx->s == nullptr || ctx->p == nullptr) {
        return;
    }
    vk_device device = ctx->p->q->handle->device.lock();
    std::lock_guard<std::mutex> lock(vk_timeline_mutex);
    auto it = vk_timelines.find(device.get());
    if (it == vk_timelines.end()) {
        return;
    }
    vk_timeline_fork & tl = *it->second;
    vk::CommandBuffer buf = ctx->s->buffer->buf;
    auto o = tl.open.find((VkCommandBuffer) buf);
    if (o == tl.open.end()) {
        return;
    }
    buf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, tl.pool, 2 * o->second + 1);
    tl.rec_end_us[o->second] = ggml_time_us();
    tl.pending.push_back(o->second);
    tl.open.erase(o);
}

// the spans whose two timestamps are there, without waiting (a buffer ended but not submitted yet stays)
static void ggml_vk_fork_timeline_read(vk_timeline_fork & tl) {
    std::vector<uint32_t> keep;
    for (const uint32_t slot : tl.pending) {
        uint64_t r[4] = {};
        const vk::Result res = tl.dev.getQueryPoolResults(tl.pool, 2 * slot, 2, sizeof(r), r, 2 * sizeof(uint64_t),
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if ((res == vk::Result::eSuccess || res == vk::Result::eNotReady) && r[1] != 0 && r[3] != 0) {
            tl.spans.push_back({ r[0] * tl.period_us, r[2] * tl.period_us, (double) tl.rec_end_us[slot] });
            tl.dev.resetQueryPool(tl.pool, 2 * slot, 2);
            tl.busy[slot] = 0;
        } else {
            keep.push_back(slot);
        }
    }
    tl.pending.swap(keep);
}

void ggml_vk_fork_timeline_collect(const vk_device & device) {
    if (!vk_timeline_any) {
        return;
    }
    std::lock_guard<std::mutex> lock(vk_timeline_mutex);
    auto it = vk_timelines.find(device.get());
    if (it != vk_timelines.end()) {
        ggml_vk_fork_timeline_read(*it->second);
    }
}

// ggml_backend_profiler_fork of the backend reg: the start of a capture drops what is left of the last one, the end
// writes the spans read back so far
static void ggml_vk_fork_profiler(bool start) {
    std::lock_guard<std::mutex> lock(vk_timeline_mutex);
    for (auto & it : vk_timelines) {
        vk_timeline_fork & tl = *it.second;
        ggml_vk_fork_timeline_read(tl);
        if (!start && !tl.spans.empty()) {
            // the end of the recording of a command buffer stands for its issue time
            std::vector<double> a, b, c;
            for (const auto & sp : tl.spans) {
                a.push_back(sp[0]);
                b.push_back(sp[1]);
                c.push_back(sp[2]);
            }
            ggml_timeline_device_spans(tl.track.c_str(), a.size(), a.data(), b.data(), c.data());
        }
        if (!start) {
            GGML_LOG_INFO("ggml_vulkan: timeline of %s: %zu gpu spans, %zu without timestamps yet, %" PRId64 " left out with the ring full\n",
                    tl.track.c_str(), tl.spans.size(), tl.pending.size(), tl.n_dropped);
        }
        tl.spans.clear();
        tl.n_dropped = 0;
    }
}

// elastic buffers (26.10): the elastic KV cache of llama (LLAMA_KV_ELASTIC, src/fork/llama-kv-elastic-fork.cpp) places its
// tensors in a buffer that reserves the range of the whole cache and binds device memory to the cells in use, per
// granule, as the CUDA one maps VMM granules (ggml-cuda/fork/elastic-buffer-fork.inc). Here a sparse buffer: the device
// needs sparseBinding and sparseResidencyBuffer and a queue family that binds sparse memory (the AMD driver of the 395
// has them, MoltenVK has not). The memory the cache does not bind stays free. The granules take their memory from slabs
// of the buffer: one granule a slab where the driver allows many allocations, several where it allows few
// (maxMemoryAllocationCount, 4096 on some Windows drivers); a slab is freed with its last granule. When the device
// memory is full, the slabs come from the shared memory (host visible, not device local) where the buffer takes it:
// on the 395 the same DRAM outside the carve-out, which the gpu reads at about the same speed (the experts of 40-47
// there cost ~1 percent TG, 26.10); GGML_VK_ELASTIC_SPILL=0 turns this off, 2 takes every slab from there (tests).
// Full is read from VK_EXT_memory_budget before a slab is allocated, a heap keeping GGML_VK_ELASTIC_MARGIN_MIB (512)
// free: the AMD driver of the 395 gives device memory past the carve-out without an error and stalls on its first use
// (the MTP head at 98.8 GB of buffers, 11.10), so a failed allocation comes too late. The compute buffers are reserved
// before the cache grows; with 2048 the chat of 27.10 went to the shared memory at 188K cells (5.0 GB of KV on the
// device), where the static cache had held all 6.9 GB. The buffer is not host visible: the tensor writes and reads go
// through the staging buffers
struct ggml_vk_elastic_fork {
    struct slab {
        vk::DeviceMemory     mem;   // null once freed, the place is taken again
        std::vector<uint8_t> used;  // per granule of the slab
        size_t               n_used = 0;
        bool                 spill  = false; // of the shared memory
    };

    vk_device               device;
    size_t                  granularity   = 0; // a multiple of the sparse page size
    size_t                  slab_granules = 1;
    uint32_t                memory_type   = 0;
    int                     spill_type    = -1; // the shared memory type, -1 for none
    bool                    spill_only    = false; // GGML_VK_ELASTIC_SPILL=2
    bool                    budget        = false; // VK_EXT_memory_budget: the room of a heap is read before a slab
    uint64_t                margin        = 0;     // bytes each heap keeps free
    bool                    device_full   = false; // the device memory had no room for a slab in this commit
    bool                    spilled       = false; // logged
    vk::MemoryAllocateFlags alloc_flags;       // the device address flag where the buffers have addresses
    std::vector<int32_t>    granule_slab;      // per granule of the buffer: its slab, -1 while unbound
    std::vector<uint32_t>   granule_piece;     // and its place in the slab
    std::vector<slab>       slabs;

    ~ggml_vk_elastic_fork() {
        for (const slab & s : slabs) {
            if (s.mem) {
                device->device.freeMemory(s.mem);
            }
        }
    }

    // a slab with room, of the device memory or the shared memory, slabs.size() for none
    size_t open_slab(bool spill) const {
        for (size_t k = 0; k < slabs.size(); ++k) {
            if (slabs[k].mem && slabs[k].spill == spill && slabs[k].n_used < slab_granules) {
                return k;
            }
        }
        return slabs.size();
    }

    // whether a slab fits the heap of the memory type within its budget, less the margin; true without the budget.
    // usage and limit: what the budget read, for the log
    bool heap_room(uint32_t type, uint64_t * usage = nullptr, uint64_t * limit = nullptr) const {
        if (!budget) {
            return true;
        }
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget_props;
        vk::PhysicalDeviceMemoryProperties2 props2;
        props2.pNext = &budget_props;
        device->physical_device.getMemoryProperties2(&props2);
        const uint32_t heap = props2.memoryProperties.memoryTypes[type].heapIndex;
        const uint64_t lim  = std::min<uint64_t>(budget_props.heapBudget[heap], props2.memoryProperties.memoryHeaps[heap].size);
        if (usage) {
            *usage = budget_props.heapUsage[heap];
        }
        if (limit) {
            *limit = lim;
        }
        return budget_props.heapUsage[heap] + slab_granules*granularity + margin <= lim;
    }

    bool alloc_slab(uint32_t type, vk::DeviceMemory & mem) const {
        vk::MemoryAllocateFlagsInfo flags_info(alloc_flags);
        try {
            mem = device->device.allocateMemory(vk::MemoryAllocateInfo(slab_granules*granularity, type, &flags_info));
        } catch (const vk::SystemError &) {
            return false;
        }
        return true;
    }

    // a place for granule i: a device slab with room, a new device slab, then the same of the shared memory; false
    // when there is no memory for one
    bool take(size_t i) {
        size_t k = open_slab(false);
        if (k == slabs.size()) {
            vk::DeviceMemory mem;
            bool spill = false;
            if (spill_only || device_full || !heap_room(memory_type) || !alloc_slab(memory_type, mem)) {
                device_full = true;
                k = spill_type >= 0 ? open_slab(true) : slabs.size();
                if (k == slabs.size()) {
                    if (spill_type < 0 || !heap_room((uint32_t) spill_type) || !alloc_slab((uint32_t) spill_type, mem)) {
                        return false;
                    }
                    spill = true;
                    if (!spilled) {
                        uint64_t usage = 0;
                        uint64_t limit = 0;
                        if (spill_only) {
                            GGML_LOG_INFO("ggml_vulkan: elastic buffer: granules from the shared memory (type %d, "
                                    "GGML_VK_ELASTIC_SPILL=2)\n", spill_type);
                        } else if (budget && !heap_room(memory_type, &usage, &limit)) {
                            GGML_LOG_INFO("ggml_vulkan: elastic buffer: the device memory is full (%.0f of %.0f MiB in "
                                    "use, %.0f kept free), granules from the shared memory (type %d)\n",
                                    usage/1048576.0, limit/1048576.0, margin/1048576.0, spill_type);
                        } else {
                            GGML_LOG_INFO("ggml_vulkan: elastic buffer: the device memory is full (an allocation failed), "
                                    "granules from the shared memory (type %d)\n", spill_type);
                        }
                        spilled = true;
                    }
                }
            }
            if (mem) {
                k = 0;
                while (k < slabs.size() && slabs[k].mem) {
                    ++k;
                }
                if (k == slabs.size()) {
                    slabs.emplace_back();
                }
                slabs[k].mem    = mem;
                slabs[k].used.assign(slab_granules, 0);
                slabs[k].n_used = 0;
                slabs[k].spill  = spill;
            }
        }
        slab & s = slabs[k];
        const size_t p = std::find(s.used.begin(), s.used.end(), 0) - s.used.begin();
        s.used[p] = 1;
        s.n_used++;
        granule_slab[i]  = (int32_t) k;
        granule_piece[i] = (uint32_t) p;
        return true;
    }

    // granule i leaves its place, after it is unbound; free_empty() then frees the slabs with no granule left
    void give(size_t i) {
        slab & s = slabs[granule_slab[i]];
        s.used[granule_piece[i]] = 0;
        s.n_used--;
        granule_slab[i] = -1;
    }

    void free_empty() {
        for (slab & s : slabs) {
            if (s.mem && s.n_used == 0) {
                device->device.freeMemory(s.mem);
                s.mem = vk::DeviceMemory();
            }
        }
    }
};

// the queue of the binds: one whose family binds sparse memory, nullptr where the device has no sparse buffers
static vk_queue * ggml_vk_fork_sparse_queue(const vk_device & device) {
    const vk::PhysicalDeviceFeatures features = device->physical_device.getFeatures();
    if (!features.sparseBinding || !features.sparseResidencyBuffer) {
        return nullptr;
    }
    const std::vector<vk::QueueFamilyProperties> families = device->physical_device.getQueueFamilyProperties();
    for (vk_queue * q : { device->compute_queue.get(), device->transfer_queue.get() }) {
        if (q != nullptr && q->queue_family_index < families.size() &&
                (families[q->queue_family_index].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
            return q;
        }
    }
    return nullptr;
}

// binds the memory of each entry (null: unbinds) to the buffer, and waits for it
static void ggml_vk_fork_sparse_bind(const vk_device & device, vk::Buffer buffer, const std::vector<vk::SparseMemoryBind> & binds) {
    if (binds.empty()) {
        return;
    }
    vk_queue * q = ggml_vk_fork_sparse_queue(device);
    GGML_ASSERT(q != nullptr);

    vk::SparseBufferMemoryBindInfo buffer_bind(buffer, (uint32_t) binds.size(), binds.data());
    vk::BindSparseInfo info;
    info.setBufferBindCount(1);
    info.setPBufferBinds(&buffer_bind);

    struct fence_guard {
        vk::Device dev;
        vk::Fence  fence;
        ~fence_guard() { dev.destroyFence(fence); }
    } guard { device->device, device->device.createFence({}) };
    {
        std::unique_lock<std::mutex> device_guard;
        if (q->handle->device_submit_mutex) {
            device_guard = std::unique_lock<std::mutex>(*q->handle->device_submit_mutex);
        }
        q->handle->lock();
        try {
            q->handle->queue.bindSparse(info, guard.fence);
        } catch (...) {
            q->handle->unlock();
            throw;
        }
        q->handle->unlock();
    }
    VK_CHECK(device->device.waitForFences({ guard.fence }, true, UINT64_MAX), "elastic bindSparse waitForFences", device);
}

// the work of the queues done, so that no command reads a granule that goes
static void ggml_vk_fork_queues_idle(const vk_device & device) {
    for (vk_queue * q : { device->compute_queue.get(), device->transfer_queue.get() }) {
        if (q == nullptr) {
            continue;
        }
        q->handle->lock();
        q->handle->queue.waitIdle();
        q->handle->unlock();
    }
}

// a buffer of `size` bytes with no memory bound yet, nullptr where the device has no sparse buffers
static ggml_backend_buffer_t ggml_backend_vk_buft_alloc_elastic_fork(ggml_backend_buffer_type_t buft, size_t size) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return nullptr;
    }
    ggml_backend_vk_buffer_type_context * bctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    vk_device device = bctx->device;
    if (bctx->shared || size == 0 || ggml_vk_fork_sparse_queue(device) == nullptr) {
        return nullptr;
    }

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                                 vk::BufferUsageFlagBits::eTransferDst;
    if (device->buffer_device_address) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    const vk::BufferCreateFlags flags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;

    // the granule: the 2 MiB that llama aligns the tensors of the cache to, in whole sparse pages
    size_t granule = 2u*1024*1024;
    size_t total   = GGML_PAD(size, granule);

    vk::Buffer buffer;
    vk::MemoryRequirements req;
    try {
        buffer = device->device.createBuffer({ flags, total, usage, vk::SharingMode::eExclusive });
        req    = device->device.getBufferMemoryRequirements(buffer);
        if (req.alignment > 0 && granule % req.alignment != 0) {
            device->device.destroyBuffer(buffer);
            buffer  = VK_NULL_HANDLE;
            granule = GGML_PAD(granule, (size_t) req.alignment);
            total   = GGML_PAD(size, granule);
            buffer  = device->device.createBuffer({ flags, total, usage, vk::SharingMode::eExclusive });
            req     = device->device.getBufferMemoryRequirements(buffer);
        }
    } catch (const vk::SystemError & e) {
        if (buffer) {
            device->device.destroyBuffer(buffer);
        }
        GGML_LOG_WARN("ggml_vulkan: no sparse buffer of %zu bytes: %s\n", size, e.what());
        return nullptr;
    }

    // no protected or lazily allocated memory (a sparse bind takes neither), no AMD device coherent or uncached memory
    // (the deviceCoherentMemory feature is not enabled)
    const vk::MemoryPropertyFlags no = vk::MemoryPropertyFlagBits::eProtected | vk::MemoryPropertyFlagBits::eLazilyAllocated |
                                       vk::MemoryPropertyFlagBits::eDeviceCoherentAMD | vk::MemoryPropertyFlagBits::eDeviceUncachedAMD;
    const vk::PhysicalDeviceMemoryProperties props = device->physical_device.getMemoryProperties();
    int memory_type = -1;
    for (uint32_t i = 0; i < props.memoryTypeCount && memory_type < 0; ++i) {
        const vk::MemoryPropertyFlags f = props.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i)) && (f & vk::MemoryPropertyFlagBits::eDeviceLocal) && !(f & no)) {
            memory_type = (int) i;
        }
    }
    if (memory_type < 0) {
        device->device.destroyBuffer(buffer);
        return nullptr;
    }
    // the shared memory for the slabs that the device memory has no room for (GGML_VK_ELASTIC_SPILL=0: none, 2: all)
    const int spill_mode = getenv("GGML_VK_ELASTIC_SPILL") != nullptr ? atoi(getenv("GGML_VK_ELASTIC_SPILL")) : 1;
    int spill_type = -1;
    if (spill_mode != 0) {
        const vk::MemoryPropertyFlags want = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        for (uint32_t i = 0; i < props.memoryTypeCount && spill_type < 0; ++i) {
            const vk::MemoryPropertyFlags f = props.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (f & want) == want && !(f & vk::MemoryPropertyFlagBits::eDeviceLocal) && !(f & no)) {
                spill_type = (int) i;
            }
        }
    }

    auto el = std::make_shared<ggml_vk_elastic_fork>();
    el->device      = device;
    el->granularity = granule;
    el->memory_type = (uint32_t) memory_type;
    el->spill_type  = spill_type;
    el->spill_only  = spill_mode == 2 && spill_type >= 0;
    el->budget      = device->idx < vk_instance.device_supports_membudget.size() && vk_instance.device_supports_membudget[device->idx];
    el->margin      = (uint64_t) std::max<int64_t>(0, getenv("GGML_VK_ELASTIC_MARGIN_MIB") != nullptr ?
                          atoll(getenv("GGML_VK_ELASTIC_MARGIN_MIB")) : 512)*1024*1024;
    if (device->buffer_device_address) {
        el->alloc_flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
    }
    el->granule_slab.assign(total/granule, -1);
    el->granule_piece.assign(total/granule, 0);

    // granules per slab: elastic buffers that fill the whole heap of their memory take at most three quarters of the
    // allocations the device allows (32 MiB slabs for a 96 GiB heap and 4096 allocations)
    {
        const uint64_t heap   = props.memoryHeaps[props.memoryTypes[memory_type].heapIndex].size;
        const uint64_t budget = std::max<uint64_t>(1, device->properties.limits.maxMemoryAllocationCount/4*3);
        el->slab_granules = (size_t) std::max<uint64_t>(1, (heap/granule + budget - 1)/budget);
    }
    GGML_LOG_INFO("ggml_vulkan: elastic buffer of %.1f MiB: granule %zu KiB, %zu per slab, memory type %d, shared type %d%s, budget %d\n",
        total/1024.0/1024.0, granule/1024, el->slab_granules, memory_type, spill_type, el->spill_only ? " (all slabs)" : "", (int) el->budget);

    vk_buffer vkb = std::make_shared<vk_buffer_struct>();
    vkb->buffer                = buffer;
    vkb->device_memory         = VK_NULL_HANDLE;
    vkb->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal; // not mapped: no direct host access
    vkb->ptr                   = nullptr;
    vkb->size                  = total;
    vkb->device                = device;
    if (device->buffer_device_address) {
        vkb->bda_addr = device->device.getBufferAddress(vk::BufferDeviceAddressInfo(buffer));
    }

    std::string name = bctx->name;
    ggml_backend_vk_buffer_context * ctx = new ggml_backend_vk_buffer_context(device, std::move(vkb), name);
    ctx->elastic = el;

    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, ctx, total);
}

// binds device memory, zeroed, to the granules of [offset, offset + size) that have none; false when the device has no
// memory left for them (the granules bound before stay bound)
static bool ggml_backend_vk_buffer_commit_fork(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    if (!ggml_backend_buffer_is_vk(buffer)) {
        return false;
    }
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    ggml_vk_elastic_fork * el = ctx->elastic.get();
    if (el == nullptr || size == 0) {
        return true;
    }

    const size_t g  = el->granularity;
    const size_t i0 = offset/g;
    const size_t i1 = std::min(el->granule_slab.size(), (offset + size + g - 1)/g);

    // the device memory freed elsewhere since the last call is tried again, once per call
    el->device_full = false;

    std::vector<vk::SparseMemoryBind> binds;
    bool ok = true;
    for (size_t i = i0; i < i1; ++i) {
        if (el->granule_slab[i] >= 0) {
            continue;
        }
        if (!el->take(i)) {
            ok = false;
            break;
        }
        binds.push_back(vk::SparseMemoryBind(i*g, g, el->slabs[el->granule_slab[i]].mem, el->granule_piece[i]*g));
    }

    try {
        ggml_vk_fork_sparse_bind(el->device, ctx->dev_buffer->buffer, binds);
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: binding %zu granules of an elastic buffer failed: %s\n", binds.size(), e.what());
        for (const auto & b : binds) {
            el->give(b.resourceOffset/g);
        }
        el->free_empty();
        return false;
    }

    // padding cells are read by the attention: no NaN in them, as the clear of an ordinary buffer
    for (size_t k = 0; k < binds.size(); ) {
        size_t j = k + 1;
        while (j < binds.size() && binds[j].resourceOffset == binds[j - 1].resourceOffset + g) {
            ++j;
        }
        ggml_vk_buffer_memset(ctx->dev_buffer, binds[k].resourceOffset, 0, (j - k)*g);
        k = j;
    }

    return ok;
}

// unbinds and frees the granules that lie whole inside [offset, offset + size)
static void ggml_backend_vk_buffer_decommit_fork(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    if (!ggml_backend_buffer_is_vk(buffer)) {
        return;
    }
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    ggml_vk_elastic_fork * el = ctx->elastic.get();
    if (el == nullptr || size == 0) {
        return;
    }

    const size_t g  = el->granularity;
    const size_t i0 = (offset + g - 1)/g;
    const size_t i1 = std::min(el->granule_slab.size(), (offset + size)/g);

    std::vector<vk::SparseMemoryBind> binds;
    for (size_t i = i0; i < i1; ++i) {
        if (el->granule_slab[i] >= 0) {
            binds.push_back(vk::SparseMemoryBind(i*g, g, vk::DeviceMemory(), 0));
        }
    }
    if (binds.empty()) {
        return;
    }

    // the granules stay bound when the unbind fails: their memory is not freed under a binding
    try {
        ggml_vk_fork_queues_idle(el->device);
        ggml_vk_fork_sparse_bind(el->device, ctx->dev_buffer->buffer, binds);
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: unbinding %zu granules of an elastic buffer failed: %s\n", binds.size(), e.what());
        return;
    }
    for (const auto & b : binds) {
        el->give(b.resourceOffset/g);
    }
    el->free_empty();
}

bool ggml_vk_fork_elastic_clear(ggml_backend_vk_buffer_context * ctx, uint8_t value) {
    ggml_vk_elastic_fork * el = ctx->elastic.get();
    if (el == nullptr) {
        return false;
    }
    const uint32_t val32 = (uint32_t) value * 0x01010101;
    const size_t   g     = el->granularity;
    for (size_t i = 0; i < el->granule_slab.size(); ) {
        if (el->granule_slab[i] < 0) {
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < el->granule_slab.size() && el->granule_slab[j] >= 0) {
            ++j;
        }
        ggml_vk_buffer_memset(ctx->dev_buffer, i*g, val32, (j - i)*g);
        i = j;
    }
    return true;
}

void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    UNUSED(reg);
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_vk_device_get_extra_bufts;
    }
    if (strcmp(name, "ggml_backend_profiler_fork") == 0) {
        return (void *) ggml_vk_fork_profiler;
    }
    if (strcmp(name, "ggml_backend_buft_alloc_elastic_fork") == 0) {
        return (void *) ggml_backend_vk_buft_alloc_elastic_fork;
    }
    if (strcmp(name, "ggml_backend_buffer_commit_fork") == 0) {
        return (void *) ggml_backend_vk_buffer_commit_fork;
    }
    if (strcmp(name, "ggml_backend_buffer_decommit_fork") == 0) {
        return (void *) ggml_backend_vk_buffer_decommit_fork;
    }
    return nullptr;
}

// GGML_VK_GRAPH_TIMING=N: the gpu time of each graph from two timestamps, against the time from one graph to the
// next and the host time in this call, printed as averages over N graphs. unlike GGML_VK_PERF_LOGGER it adds no
// barrier and no wait, so the gpu idle time of a decode step shows
static int ggml_vk_graph_timing() {
    static const int graph_timing = [] {
        const char * env = getenv("GGML_VK_GRAPH_TIMING");
        return env ? std::max(1, atoi(env)) : 0;
    }();
    return graph_timing;
}

int64_t ggml_vk_fork_graph_timing_begin(ggml_backend_vk_context * ctx) {
    const int graph_timing = ggml_vk_graph_timing();
    const int64_t t_enter_us = graph_timing ? ggml_time_us() : 0;
    if (graph_timing && !vk_perf_logger_enabled) {
        if (!ctx->gt_pool) {
            vk::QueryPoolCreateInfo info;
            info.queryType  = vk::QueryType::eTimestamp;
            info.queryCount = 2;
            ctx->gt_pool = ctx->device->device.createQueryPool(info);
        }
        if (ctx->gt_pending) {
            uint64_t ts[2] = {};
            VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->gt_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "graph timing results", ctx->device);
            ctx->gt_gpu_ms    += 1e-6 * (double) (ts[1] - ts[0]) * ctx->device->properties.limits.timestampPeriod;
            ctx->gt_period_ms += 1e-3 * (double) (t_enter_us - ctx->gt_enter_us);
            ctx->gt_pending    = false;
            if (++ctx->gt_n >= (uint32_t) graph_timing) {
                const double n = ctx->gt_n;
                fprintf(stderr, "ggml_vulkan: graph timing over %u graphs: one every %.2f ms, gpu %.2f ms, in graph_compute %.2f ms, gpu idle %.2f ms (%.0f%%)\n",
                    ctx->gt_n, ctx->gt_period_ms / n, ctx->gt_gpu_ms / n, ctx->gt_host_ms / n,
                    (ctx->gt_period_ms - ctx->gt_gpu_ms) / n, 100.0 * (ctx->gt_period_ms - ctx->gt_gpu_ms) / ctx->gt_period_ms);
                ctx->gt_n = 0;
                ctx->gt_gpu_ms = ctx->gt_host_ms = ctx->gt_period_ms = 0;
            }
        }
        ctx->gt_enter_us = t_enter_us;
        ctx->device->device.resetQueryPool(ctx->gt_pool, 0, 2);
        vk_context gt_ctx = ggml_vk_get_compute_ctx(ctx);
        gt_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ctx->gt_pool, 0);
        ctx->gt_recording = true;
    }
    return t_enter_us;
}

void ggml_vk_fork_graph_timing_end(ggml_backend_vk_context * ctx, const vk_context & compute_ctx, bool last_node) {
    if (last_node && ctx->gt_recording) {
        compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, ctx->gt_pool, 1);
        ctx->gt_recording = false;
        ctx->gt_pending   = true;
    }
}

void ggml_vk_fork_graph_timing_finish(ggml_backend_vk_context * ctx, int64_t t_enter_us) {
    if (ggml_vk_graph_timing() && ctx->gt_pending) {
        ctx->gt_host_ms += 1e-3 * (double) (ggml_time_us() - t_enter_us);
    }
}
