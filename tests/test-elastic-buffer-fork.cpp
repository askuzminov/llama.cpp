// the elastic buffers of the backends, which the elastic KV cache of llama takes (LLAMA_KV_ELASTIC,
// src/fork/llama-kv-elastic-fork.cpp) through the proc addresses of a backend registry: a range reserved whole, with
// device memory mapped (CUDA, VMM granules) or bound (Vulkan, a sparse buffer) to the granules in use. on every GPU
// device that has them: memory for some granules, which reads zero; tensors written and read back there and a graph
// computed on them; granules taken and given back while a graph runs on the others; granules given back and taken
// again, zeroed, while the others keep their data; a clear of the bound granules only. on Vulkan all of it once more
// with every granule from the shared memory (GGML_VK_ELASTIC_SPILL=2), where the KV goes when the device memory is
// full. a device without them is skipped (MoltenVK has no sparse buffers, the CPU no elastic buffers);
// TEST_ELASTIC_REQUIRE=1 (01-check on the machines, _config.bat) fails the test when no device was tested

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

typedef ggml_backend_buffer_t (*alloc_elastic_t)(ggml_backend_buffer_type_t buft, size_t size);
typedef bool (*commit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);
typedef void (*decommit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);

static int n_fail = 0;

static void check(bool ok, const char * what, const std::string & where, int line) {
    if (!ok) {
        fprintf(stderr, "%s, line %d: %s\n", where.c_str(), line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, where, __LINE__)

static const size_t G = 2u*1024*1024; // the granule llama aligns the tensors of the cache to

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static bool all_zero(const std::vector<float> & v) {
    for (float x : v) {
        if (x != 0.0f) {
            return false;
        }
    }
    return true;
}

static bool equal_sum(const std::vector<float> & c, const std::vector<float> & x, const std::vector<float> & y) {
    for (size_t k = 0; k < c.size(); ++k) {
        if (c[k] != x[k] + y[k]) {
            return false;
        }
    }
    return true;
}

// c = x + y over a graph of its own, computed now or started (async) and left running; nullptr when it fails
static ggml_gallocr_t add_into(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * x, ggml_tensor * y, ggml_tensor * c, bool async) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, x, y), c));
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        return nullptr;
    }
    const ggml_status st = async ? ggml_backend_graph_compute_async(backend, gf) : ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        return nullptr;
    }
    return galloc;
}

// false when the device gives no elastic buffer
static bool test_device(ggml_backend_dev_t dev, alloc_elastic_t alloc, commit_t commit, decommit_t decommit, const char * pass) {
    const std::string where = std::string(ggml_backend_dev_name(dev)) + (pass[0] ? std::string(" ") + pass : std::string());

    // 16 granules reserved; a takes granules 0-1, b 2-3, c 4-5
    ggml_backend_buffer_t buf = alloc(ggml_backend_dev_buffer_type(dev), 16*G);
    if (buf == nullptr) {
        return false;
    }
    CHECK(ggml_backend_buffer_get_size(buf) >= 16*G);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "%s: the backend did not start\n", where.c_str());
        n_fail++;
        ggml_backend_buffer_free(buf);
        return true;
    }

    ggml_init_params params = { /*.mem_size =*/ 32*ggml_tensor_overhead() + 4*ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(params);

    const int64_t n = (int64_t) (2*G/sizeof(float));
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    char * base = (char *) ggml_backend_buffer_get_base(buf);
    CHECK(ggml_backend_tensor_alloc(buf, a, base + 0*G) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_tensor_alloc(buf, b, base + 2*G) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_tensor_alloc(buf, c, base + 4*G) == GGML_STATUS_SUCCESS);

    // memory for the three, zeroed as a fresh buffer is cleared
    CHECK(commit(buf, 0, 6*G));
    CHECK(all_zero(get(a)) && all_zero(get(b)) && all_zero(get(c)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> va(n), vb(n);
    for (int64_t k = 0; k < n; ++k) {
        va[k] = dist(rng);
        vb[k] = dist(rng);
    }
    ggml_backend_tensor_set(a, va.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, vb.data(), 0, ggml_nbytes(b));
    CHECK(get(a) == va && get(b) == vb);

    // the shaders read and write the bound memory: c = a + b
    {
        ggml_gallocr_t galloc = add_into(backend, ctx, a, b, c, false);
        CHECK(galloc != nullptr);
        CHECK(equal_sum(get(c), va, vb));
        ggml_gallocr_free(galloc);
    }

    // granules taken and given back while a graph computes c = a + a on the others
    CHECK(commit(buf, 14*G, 2*G));
    {
        ggml_gallocr_t galloc = add_into(backend, ctx, a, a, c, true);
        CHECK(galloc != nullptr);
        CHECK(commit(buf, 6*G, 2*G));
        decommit(buf, 14*G, 2*G);
        ggml_backend_synchronize(backend);
        CHECK(equal_sum(get(c), va, va));
        ggml_gallocr_free(galloc);
    }

    // b and c give their granules back and take them again: zeroed, while a keeps its data
    decommit(buf, 2*G, 4*G);
    CHECK(commit(buf, 2*G, 4*G));
    CHECK(all_zero(get(b)) && all_zero(get(c)));
    CHECK(get(a) == va);

    // a range with its granules bound is no change; one past the end is cut at the end
    CHECK(commit(buf, 0, 2*G));
    CHECK(get(a) == va);
    CHECK(commit(buf, 14*G, 4*G));

    // the clear touches the bound granules only
    ggml_backend_buffer_clear(buf, 0);
    CHECK(all_zero(get(a)));

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    // what a commit takes from the free memory of the device and a decommit gives back: for the log, not checked, the
    // drivers report the free memory in steps of their own
    ggml_backend_buffer_t big = alloc(ggml_backend_dev_buffer_type(dev), 256*G);
    if (big != nullptr) {
        size_t free0 = 0, free1 = 0, free2 = 0, total = 0;
        ggml_backend_dev_memory(dev, &free0, &total);
        CHECK(commit(big, 0, 128*G));
        ggml_backend_dev_memory(dev, &free1, &total);
        decommit(big, 0, 128*G);
        ggml_backend_dev_memory(dev, &free2, &total);
        printf("%s: free memory %zu MiB, %zu MiB after a commit of %zu MiB, %zu MiB after the decommit\n", where.c_str(),
                free0/1024/1024, free1/1024/1024, 128*G/1024/1024, free2/1024/1024);
        ggml_backend_buffer_free(big);
    }

    ggml_backend_free(backend);

    printf("%s: elastic buffer ok\n", where.c_str());
    return true;
}

int main() {
    ggml_backend_load_all();

    int n_tested = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto alloc    = reg ? (alloc_elastic_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buft_alloc_elastic_fork") : nullptr;
        auto commit   = reg ? (commit_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_commit_fork")      : nullptr;
        auto decommit = reg ? (decommit_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_decommit_fork")    : nullptr;
        if (!alloc || !commit || !decommit) {
            printf("%s: no elastic buffers in its backend, skipped\n", ggml_backend_dev_name(dev));
            continue;
        }

        if (!test_device(dev, alloc, commit, decommit, "")) {
            printf("%s: the device gives no elastic buffer (no VMM, no sparse buffers), skipped\n", ggml_backend_dev_name(dev));
            continue;
        }
        n_tested++;

        // Vulkan: every granule from the shared memory (the "elastic buffer" line of the log says whether the buffer
        // takes it; where it does not, this pass runs on the device memory again)
        if (strncmp(ggml_backend_reg_name(reg), "Vulkan", 6) == 0) {
            const char * prev = getenv("GGML_VK_ELASTIC_SPILL");
            const std::string keep = prev ? prev : "";
            set_env("GGML_VK_ELASTIC_SPILL", "2");
            test_device(dev, alloc, commit, decommit, "(shared memory)");
            set_env("GGML_VK_ELASTIC_SPILL", prev ? keep.c_str() : nullptr);
        }
    }

    const char * require = getenv("TEST_ELASTIC_REQUIRE");
    if (n_tested == 0 && require != nullptr && atoi(require) != 0) {
        fprintf(stderr, "no device gave an elastic buffer, and TEST_ELASTIC_REQUIRE is set\n");
        n_fail++;
    }

    printf("test-elastic-buffer-fork: %d devices tested%s\n", n_tested, n_fail ? ", FAILED" : "");
    return n_fail == 0 ? 0 : 1;
}
