// fork: the temperature, power and fan of the GPUs of this machine for the dashboard of a pool (GET /pool/node "gpus",
// server-pool-fork.h). no vendor library:
// - Windows: the performance data a WDDM 2.4+ driver gives the kernel graphics interface (D3DKMTQueryAdapterInfo with
//   KMTQAITYPE_ADAPTERPERFDATA, what Task Manager shows), from gdi32.dll at run time; the structures are declared here
//   as in d3dkmthk.h of the Windows SDK. a driver that does not report a value leaves it 0, which is left out (Task
//   Manager shows the temperature of dedicated GPUs only, an integrated one may report none)
// - Linux: hwmon of the DRM cards (temp1_input, power1_average or power1_input, fan1_input)
// - elsewhere: none
// read at most every 2 s, the polls of the routers of a pool share it

#include "server-pool-fork.h"

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <filesystem>
#include <fstream>
#endif

#if defined(_WIN32)

namespace {

typedef UINT kmt_handle;

struct kmt_adapter_info { // D3DKMT_ADAPTERINFO
    kmt_handle hAdapter;
    LUID       AdapterLuid;
    ULONG      NumOfSources;
    BOOL       bPrecisePresentRegionsPreferred;
};

struct kmt_enum_adapters2 { // D3DKMT_ENUMADAPTERS2
    ULONG              NumAdapters;
    kmt_adapter_info * pAdapters;
};

struct kmt_query_adapter_info { // D3DKMT_QUERYADAPTERINFO
    kmt_handle hAdapter;
    UINT       Type; // KMTQUERYADAPTERINFOTYPE
    void *     pPrivateDriverData;
    UINT       PrivateDriverDataSize;
};

struct kmt_close_adapter { // D3DKMT_CLOSEADAPTER
    kmt_handle hAdapter;
};

struct kmt_registry_info { // D3DKMT_ADAPTERREGISTRYINFO
    WCHAR AdapterString[MAX_PATH];
    WCHAR BiosString[MAX_PATH];
    WCHAR DacType[MAX_PATH];
    WCHAR ChipType[MAX_PATH];
};

struct kmt_perf_data { // D3DKMT_ADAPTER_PERFDATA
    UINT32    PhysicalAdapterIndex; // in
    ULONGLONG MemoryFrequency;
    ULONGLONG MaxMemoryFrequency;
    ULONGLONG MaxMemoryFrequencyOC;
    ULONGLONG MemoryBandwidth;
    ULONGLONG PCIEBandwidth;
    ULONG     FanRPM;
    ULONG     Power;       // tenths of a percent of the power limit
    ULONG     Temperature; // tenths of a degree Celsius
    UCHAR     PowerStateOverride;
};

constexpr UINT KMT_ADAPTERREGISTRYINFO = 8;
constexpr UINT KMT_ADAPTERPERFDATA     = 62;

typedef LONG (APIENTRY * kmt_enum2_fn)(kmt_enum_adapters2 *);
typedef LONG (APIENTRY * kmt_query_fn)(kmt_query_adapter_info *);
typedef LONG (APIENTRY * kmt_close_fn)(kmt_close_adapter *);

std::string kmt_utf8(const WCHAR * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) {
        return std::string();
    }
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

} // namespace

static json gpu_sensors_read() {
    json out = json::array();
    static HMODULE gdi = LoadLibraryW(L"gdi32.dll");
    if (gdi == nullptr) {
        return out;
    }
    const auto enum2 = (kmt_enum2_fn) (void *) GetProcAddress(gdi, "D3DKMTEnumAdapters2");
    const auto query = (kmt_query_fn) (void *) GetProcAddress(gdi, "D3DKMTQueryAdapterInfo");
    const auto close = (kmt_close_fn) (void *) GetProcAddress(gdi, "D3DKMTCloseAdapter");
    if (enum2 == nullptr || query == nullptr || close == nullptr) {
        return out;
    }
    kmt_enum_adapters2 e = {};
    if (enum2(&e) != 0 || e.NumAdapters == 0) {
        return out;
    }
    std::vector<kmt_adapter_info> adapters(e.NumAdapters);
    e.pAdapters = adapters.data();
    if (enum2(&e) != 0) {
        return out;
    }
    for (ULONG i = 0; i < e.NumAdapters && i < adapters.size(); ++i) {
        kmt_registry_info reg = {};
        kmt_query_adapter_info q = {adapters[i].hAdapter, KMT_ADAPTERREGISTRYINFO, &reg, (UINT) sizeof(reg)};
        std::string name = query(&q) == 0 ? kmt_utf8(reg.AdapterString) : std::string();

        kmt_perf_data perf = {};
        q = {adapters[i].hAdapter, KMT_ADAPTERPERFDATA, &perf, (UINT) sizeof(perf)};
        const bool ok = query(&q) == 0;

        kmt_close_adapter c = {adapters[i].hAdapter};
        close(&c);

        if (!ok || (perf.Temperature == 0 && perf.Power == 0 && perf.FanRPM == 0)) {
            continue;
        }
        json g = {{"name", name.empty() ? "GPU " + std::to_string(i) : name}};
        if (perf.Temperature > 0) {
            g["temp_c"] = perf.Temperature / 10.0;
        }
        if (perf.Power > 0) {
            g["power_pct"] = perf.Power / 10.0;
        }
        if (perf.FanRPM > 0) {
            g["fan_rpm"] = perf.FanRPM;
        }
        out.push_back(g);
    }
    return out;
}

#elif defined(__linux__)

static bool read_num(const std::filesystem::path & p, double & v) {
    std::ifstream f(p);
    return static_cast<bool>(f >> v);
}

static json gpu_sensors_read() {
    namespace fs = std::filesystem;
    json out = json::array();
    std::error_code ec;
    for (const auto & card : fs::directory_iterator("/sys/class/drm", ec)) {
        const std::string cn = card.path().filename().string();
        if (cn.rfind("card", 0) != 0 || cn.find('-') != std::string::npos) {
            continue; // card0-DP-1 and the like are connectors
        }
        std::error_code ec2;
        for (const auto & hw : fs::directory_iterator(card.path() / "device" / "hwmon", ec2)) {
            double t = 0.0;
            double p = 0.0;
            double fan = 0.0;
            const bool has_t = read_num(hw.path() / "temp1_input", t);
            const bool has_p = read_num(hw.path() / "power1_average", p) || read_num(hw.path() / "power1_input", p);
            const bool has_f = read_num(hw.path() / "fan1_input", fan);
            if (!has_t && !has_p && !has_f) {
                continue;
            }
            std::string driver;
            std::ifstream(hw.path() / "name") >> driver;
            json g = {{"name", (driver.empty() ? std::string("gpu") : driver) + " " + cn}};
            if (has_t) {
                g["temp_c"] = t / 1000.0; // millidegrees
            }
            if (has_p) {
                g["power_w"] = p / 1e6; // microwatts
            }
            if (has_f && fan > 0) {
                g["fan_rpm"] = fan;
            }
            out.push_back(g);
        }
    }
    return out;
}

#else

static json gpu_sensors_read() {
    return json::array();
}

#endif

json server_gpu_sensors_fork() {
    static std::mutex mu;
    static json       last = json::array();
    static auto       t_last = std::chrono::steady_clock::time_point();
    std::lock_guard<std::mutex> lk(mu);
    const auto now = std::chrono::steady_clock::now();
    if (t_last.time_since_epoch().count() == 0 || now - t_last >= std::chrono::seconds(2)) {
        last   = gpu_sensors_read();
        t_last = now;
    }
    return last;
}
