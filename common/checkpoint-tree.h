#pragma once

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

// A tree of context checkpoints for a single slot.
//
// Each node owns a checkpoint payload (common_prompt_checkpoint: base / delta / replay) and the
// token "span" it appended relative to its parent. The full token sequence covered by a node is
// the concatenation of the spans along the root->node path. Storing branches as a tree lets an
// agent that explores several continuations from a shared prefix return to any of them without
// recomputing the prefix - unlike a single linear chain, which can only remember one branch.
//
// This component is intentionally free of any llama/model dependency beyond the payload type, so
// its logic (matching, branching, eviction) is fully unit-testable without loading a model.
struct common_checkpoint_tree {
    struct node {
        int32_t id     = -1;
        int32_t parent = -1;                  // -1 = root (a BASE checkpoint)
        std::vector<llama_token> span;        // tokens this node added: (parent.pos_max, this.pos_max]
        common_prompt_checkpoint cp;          // serialized checkpoint payload
        int64_t last_used = 0;                // LRU tick; the whole root->node path is bumped on match
    };

    std::vector<node> nodes;
    int32_t next_id = 0;
    int64_t tick    = 0;

    // --- lookup helpers ---------------------------------------------------

    int32_t index_of(int32_t id) const {
        const auto it = idx.find(id);
        return it == idx.end() ? -1 : it->second;
    }

    const node * get(int32_t id) const {
        const int32_t i = index_of(id);
        return i < 0 ? nullptr : &nodes[i];
    }

    bool has_children(int32_t id) const {
        for (const node & n : nodes) {
            if (n.parent == id) {
                return true;
            }
        }
        return false;
    }

    // Number of direct children of `id`.
    int n_children(int32_t id) const {
        int c = 0;
        for (const node & n : nodes) {
            if (n.parent == id) {
                ++c;
            }
        }
        return c;
    }

    // Index (into `nodes`) of the first direct child of `id`, or -1 if it has none.
    int32_t first_child_index(int32_t id) const {
        for (int32_t j = 0; j < (int32_t) nodes.size(); ++j) {
            if (nodes[j].parent == id) {
                return j;
            }
        }
        return -1;
    }

    size_t count() const { return nodes.size(); }

    void clear() {
        nodes.clear();
        idx.clear();
        next_id = 0;
        tick    = 0;
    }

    // --- mutation ---------------------------------------------------------

    // Add a checkpoint as a child of `parent` (-1 for a new root). Returns the new node id.
    int32_t add(int32_t parent, std::vector<llama_token> span, common_prompt_checkpoint cp) {
        node n;
        n.id        = next_id++;
        n.parent    = parent;
        n.span      = std::move(span);
        n.cp        = std::move(cp);
        n.last_used = ++tick;
        nodes.push_back(std::move(n));
        idx[nodes.back().id] = (int32_t) nodes.size() - 1;
        return nodes.back().id;
    }

    // --- queries ----------------------------------------------------------

    // root->id path (ids, root first). Empty if id is invalid.
    std::vector<int32_t> path_to(int32_t id) const {
        std::vector<int32_t> path;
        int32_t cur = id;
        while (cur >= 0) {
            const int32_t i = index_of(cur);
            if (i < 0) {
                return {}; // broken link -> treat as invalid
            }
            path.push_back(cur);
            cur = nodes[i].parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // Find the deepest node whose full root->node token path is a prefix of `tokens`.
    // Returns the node id (-1 if none matches) and sets `matched_tokens` to that prefix length.
    int32_t find_best(const std::vector<llama_token> & tokens, size_t & matched_tokens) const {
        int32_t best_id  = -1;
        size_t  best_len = 0;
        for (const node & n : nodes) {
            if (n.parent == -1) {
                match_rec(n.id, tokens, 0, best_id, best_len);
            }
        }
        matched_tokens = best_len;
        return best_id;
    }

    // --- bookkeeping ------------------------------------------------------

    // Mark the whole root->id path as most-recently-used.
    void touch(int32_t id) {
        const int64_t now = ++tick;
        for (int32_t nid : path_to(id)) {
            const int32_t i = index_of(nid);
            if (i >= 0) {
                nodes[i].last_used = now;
            }
        }
    }

    size_t size_bytes() const {
        size_t total = 0;
        for (const node & n : nodes) {
            total += n.cp.size();
        }
        return total;
    }

    // Evict least-recently-used nodes until size_bytes() <= budget (budget 0 = no limit).
    // Lossless w.r.t. every surviving node's restored state:
    //   - a leaf is simply dropped;
    //   - a single-child node is "thinned": it is removed and its child re-parented to the
    //     grandparent, with their token spans merged, so the child's full root->node token path is
    //     preserved. Only the ability to restore *at* the removed node's position is lost.
    // A single-child node can be thinned ONLY when its child is self-contained (base_pos < 0): a
    // delta child depends on its parent's state, so removing the parent would break it (see
    // is_removable). Branch points (>1 child) and `protect` are never removed. Because thinning
    // can remove interior nodes too, eviction follows true LRU instead of peeling only from the
    // leaves. Returns the number of bytes freed.
    size_t evict_to_budget(size_t budget, int32_t protect = -1) {
        if (budget == 0) {
            return 0;
        }

        size_t total = size_bytes();
        size_t freed = 0;
        while (total > budget) {
            const int32_t lru = find_lru(protect);
            if (lru < 0) {
                break; // nothing removable (all remaining are protected / branch points)
            }
            const size_t n_bytes = nodes[lru].cp.size();
            freed += n_bytes;
            total -= std::min(total, n_bytes);
            remove_node(nodes[lru].id);
        }
        return freed;
    }

    // Evict least-recently-used nodes until count() <= max_nodes (0 = no limit). Same lossless
    // rules as evict_to_budget. Returns the number of bytes freed.
    size_t evict_to_count(size_t max_nodes, int32_t protect = -1) {
        if (max_nodes == 0) {
            return 0;
        }

        size_t freed = 0;
        while (nodes.size() > max_nodes) {
            const int32_t lru = find_lru(protect);
            if (lru < 0) {
                break;
            }
            freed += nodes[lru].cp.size();
            remove_node(nodes[lru].id);
        }
        return freed;
    }

private:
    std::unordered_map<int32_t, int32_t> idx; // node id -> index into `nodes`

    // Index of the least-recently-used removable node, or -1 if there is none.
    int32_t find_lru(int32_t protect) const {
        int32_t lru = -1;
        for (int32_t i = 0; i < (int32_t) nodes.size(); ++i) {
            if (nodes[i].id == protect) {
                continue; // never evict the protected node
            }
            if (!is_removable(nodes[i].id)) {
                continue;
            }
            if (lru < 0 || nodes[i].last_used < nodes[lru].last_used) {
                lru = i;
            }
        }
        return lru;
    }

    void reindex() {
        idx.clear();
        idx.reserve(nodes.size());
        for (int32_t i = 0; i < (int32_t) nodes.size(); ++i) {
            idx[nodes[i].id] = i;
        }
    }

    // A node can be removed losslessly if it is a leaf, or a single-child node whose child is
    // self-contained (the child does not depend on this node's state). Branch points cannot.
    bool is_removable(int32_t id) const {
        const int nc = n_children(id);
        if (nc == 0) {
            return true; // leaf
        }
        if (nc == 1) {
            const int32_t ci = first_child_index(id);
            return ci >= 0 && !nodes[ci].cp.is_delta(); // thin only if the child is self-contained
        }
        return false; // branch point
    }

    // Remove a leaf, or thin a single-child node: re-parent its child to the grandparent and
    // prepend this node's span to the child's. Caller must ensure is_removable(id).
    void remove_node(int32_t id) {
        const int32_t i = index_of(id);
        if (i < 0) {
            return;
        }
        const int32_t ci = first_child_index(id);
        if (ci >= 0) {
            std::vector<llama_token> merged = nodes[i].span;
            merged.insert(merged.end(), nodes[ci].span.begin(), nodes[ci].span.end());
            nodes[ci].span   = std::move(merged);
            nodes[ci].parent = nodes[i].parent;
        }
        nodes.erase(nodes.begin() + i);
        reindex();
    }

    // Recursively match node `id`'s span against tokens starting at `offset`. The node is a valid
    // restore point only if its span matches in full (the checkpoint sits at node.pos_max); a
    // partial/divergent match stops the descent, leaving the deepest fully-matched ancestor as the
    // candidate.
    void match_rec(int32_t id, const std::vector<llama_token> & tokens, size_t offset,
                   int32_t & best_id, size_t & best_len) const {
        const int32_t i = index_of(id);
        if (i < 0) {
            return;
        }
        const node & n = nodes[i];

        if (offset + n.span.size() > tokens.size()) {
            return; // request is too short to cover this node's span in full
        }
        for (size_t k = 0; k < n.span.size(); ++k) {
            if (tokens[offset + k] != n.span[k]) {
                return; // request diverges inside this node's span
            }
        }

        const size_t new_offset = offset + n.span.size();
        if (new_offset > best_len) {
            best_len = new_offset;
            best_id  = id;
        }

        for (const node & c : nodes) {
            if (c.parent == id) {
                match_rec(c.id, tokens, new_offset, best_id, best_len);
            }
        }
    }
};
