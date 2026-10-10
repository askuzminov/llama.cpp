// device setup, tile overrides, the shared buffer type and graph timing of the fork (see ggml-vulkan-fork.h)

#include "ggml-vulkan-fork.h"

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
    // GGML_VK_MMID_F16REG=1: the experts' prefill matmuls on mul_mmid_f16reg.comp (f16 coopmat, the codes decoded into
    // the A fragments) where coopmat_a_probe.comp finds the A layout it takes; =emulate: the same algorithm with
    // subgroup shuffles on any device with 32-invocation subgroups (tests). unset or 0: off
    device->mmid_f16reg_mode = 0;
    const char * env = getenv("GGML_VK_MMID_F16REG");
    if (env == nullptr || strcmp(env, "0") == 0) {
        return;
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
        return;
    }
    device->mmid_f16reg_mode = 1;
    GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID f16reg on (coopmat A layout probe passed)\n");
#else
    GGML_LOG_WARN("ggml_vulkan: GGML_VK_MMID_F16REG needs a build with coopmat shaders\n");
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

void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    UNUSED(reg);
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_vk_device_get_extra_bufts;
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
