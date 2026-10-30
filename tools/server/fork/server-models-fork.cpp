// fork: router memory wait, the router side (server_models::mem_wait_env); the child side is in server-role-fork.cpp

#include "server-common.h"
#include "server-models.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <string>

// the driver can release the memory of an exited child some time after the process ended.
// a child spawned in this time after an exit first waits for that memory, see mem_wait_env
#define MEM_WAIT_MS 120000

//
// server_models
//

std::string server_models::mem_wait_env() const {
    const int64_t left = t_last_exit + MEM_WAIT_MS - ggml_time_ms();
    if (t_last_exit == 0 || left <= 0) {
        return "";
    }
    json need = json::object();
    for (const auto & [dev, m] : mem_last.items()) {
        const int64_t free  = json_value(m, "free",  (int64_t) 0);
        const int64_t used  = json_value(m, "used",  (int64_t) 0);
        const int64_t total = json_value(m, "total", (int64_t) 0);
        // other programs also change the free memory, so only a device the last child filled noticeably is waited for
        const int64_t slack = std::max<int64_t>(1ll << 30, total / 20);
        if (used >= slack) {
            // room for the same load, but not more than was free before it
            need[dev] = std::min(free - slack, used + slack);
        }
    }
    if (need.empty()) {
        return "";
    }
    return safe_json_to_str({{"ms", left}, {"need", need}});
}
