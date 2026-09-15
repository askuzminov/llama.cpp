#include "checkpoint-tree.h"

#include <cstdio>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

static common_prompt_checkpoint mk(size_t bytes) {
    common_prompt_checkpoint cp;
    cp.data_tgt.resize(bytes);
    return cp; // self-contained (base_pos = -1)
}

static common_prompt_checkpoint mkd(size_t bytes, llama_pos base_pos) {
    common_prompt_checkpoint cp;
    cp.data_tgt.resize(bytes);
    cp.base_pos = base_pos; // delta (depends on its parent's state)
    return cp;
}

int main() {
    // 1. empty tree
    {
        common_checkpoint_tree t;
        size_t m = 123;
        CHECK(t.find_best({1, 2, 3}, m) == -1);
        CHECK(m == 0);
        CHECK(t.size_bytes() == 0);
        CHECK(t.evict_to_budget(0) == 0);
    }

    // 2. single root: full-span prefix match only
    {
        common_checkpoint_tree t;
        const int32_t r = t.add(-1, {1, 2, 3}, mk(10));
        size_t m = 0;

        CHECK(t.find_best({1, 2, 3, 4, 5}, m) == r && m == 3); // covered + extra
        CHECK(t.find_best({1, 2, 3}, m) == r && m == 3);       // exact
        CHECK(t.find_best({1, 2}, m) == -1 && m == 0);         // request shorter than span
        CHECK(t.find_best({9, 9, 9}, m) == -1 && m == 0);      // diverges immediately
        CHECK(t.find_best({1, 2, 9}, m) == -1 && m == 0);      // diverges inside span
    }

    // 3. branching: two children sharing a common first token
    {
        common_checkpoint_tree t;
        const int32_t root = t.add(-1, {1, 2},    mk(100)); // shared prefix
        const int32_t a    = t.add(root, {3, 4},  mk(10));  // branch A
        const int32_t b    = t.add(root, {3, 5},  mk(10));  // branch B
        size_t m = 0;

        CHECK(t.find_best({1, 2, 3, 4, 9}, m) == a && m == 4);
        CHECK(t.find_best({1, 2, 3, 5, 9}, m) == b && m == 4);
        CHECK(t.find_best({1, 2, 3, 6},    m) == root && m == 2); // diverges past shared token
        CHECK(t.find_best({1, 2},          m) == root && m == 2);

        // deeper chain on branch A
        const int32_t a2 = t.add(a, {7}, mk(10));
        CHECK(t.find_best({1, 2, 3, 4, 7, 8}, m) == a2 && m == 5);
        CHECK(t.find_best({1, 2, 3, 4, 9},    m) == a  && m == 4); // A2 span diverges -> fall back to A

        const std::vector<int32_t> path = t.path_to(a2);
        CHECK((path == std::vector<int32_t>{root, a, a2}));

        CHECK(t.has_children(root));
        CHECK(t.has_children(a));
        CHECK(!t.has_children(b));
        CHECK(!t.has_children(a2));
    }

    // 4. true-LRU eviction with thinning: a single-child interior node can be removed (not only
    //    leaves), as long as its child is self-contained; branch points are protected.
    {
        common_checkpoint_tree t;
        const int32_t root = t.add(-1,   {1, 2}, mk(100)); // branch point (2 children)
        const int32_t a    = t.add(root, {3},    mk(10));  // single-child interior
        const int32_t b    = t.add(root, {4},    mk(10));  // leaf
        const int32_t a2   = t.add(a,    {5},    mk(10));  // leaf, self-contained
        CHECK(t.size_bytes() == 130);

        // touch B so the LRU order is: a (oldest) < a2 < root = b (newest)
        t.touch(b);

        // a is the LRU *removable* node (root is a branch point). It gets THINNED: a2 re-parents
        // to root and inherits a's span, so a2's full token path {1,2,3,5} is preserved.
        const size_t freed = t.evict_to_budget(125);
        CHECK(freed == 10);
        CHECK(t.get(a) == nullptr);                     // LRU interior node thinned away
        CHECK(t.get(root) && t.get(b) && t.get(a2));    // branch point + leaves survive
        CHECK(t.size_bytes() == 120);

        size_t m = 0;
        CHECK(t.find_best({1, 2, 3, 5, 9}, m) == a2 && m == 4); // merged span still matches
        CHECK((t.path_to(a2) == std::vector<int32_t>{root, a2}));

        // tighten hard: with protect unset the whole tree drains to empty
        t.evict_to_budget(1);
        CHECK(t.count() == 0);
    }

    // 5. thinning is a no-op on restore data: the surviving child keeps its own payload, and a
    //    protected node is never removed even when it is the LRU.
    {
        common_checkpoint_tree t;
        const int32_t x = t.add(-1, {1}, mk(50));
        const int32_t y = t.add(x,  {2}, mk(50));
        const int32_t z = t.add(y,  {3}, mk(50)); // linear chain x->y->z, all self-contained
        t.touch(z);

        // protect z and evict hard: x and y (interior) must be thinned away but z kept intact
        t.evict_to_budget(50, z);
        CHECK(t.get(z) != nullptr);
        CHECK(t.get(z)->cp.data_tgt.size() == 50); // payload untouched by thinning
        CHECK(t.count() == 1);
        size_t m = 0;
        CHECK(t.find_best({1, 2, 3, 7}, m) == z && m == 3); // z now covers the whole path
    }

    // 6. a delta child pins its parent: a single-child node whose child is a delta must NOT be
    //    thinned (the delta depends on the parent's state), even if the parent is the LRU.
    {
        common_checkpoint_tree t;
        const int32_t b0 = t.add(-1, {1, 2}, mk(10));       // BASE (self-contained)
        const int32_t d1 = t.add(b0, {3},    mkd(10, 1));   // delta on b0
        const int32_t d2 = t.add(d1, {4},    mkd(10, 2));   // delta on d1 (leaf)
        CHECK(t.size_bytes() == 30);

        // b0 is the LRU, but its only child d1 is a delta -> b0 is not removable. Only the leaf d2
        // may go first.
        t.evict_to_budget(25);
        CHECK(t.get(d2) == nullptr);         // leaf removed
        CHECK(t.get(b0) && t.get(d1));       // delta chain root + interior kept (not thinned)
        CHECK(t.size_bytes() == 20);
    }

    // 7. multiple independent roots (different prefixes)
    {
        common_checkpoint_tree t;
        const int32_t r1 = t.add(-1, {1, 1}, mk(10));
        const int32_t r2 = t.add(-1, {2, 2}, mk(10));
        size_t m = 0;
        CHECK(t.find_best({1, 1, 9}, m) == r1 && m == 2);
        CHECK(t.find_best({2, 2, 9}, m) == r2 && m == 2);
        CHECK(t.find_best({3, 3},    m) == -1 && m == 0);
    }

    printf("checkpoint tree tests passed\n");
    return 0;
}
