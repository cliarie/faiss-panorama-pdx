#ifndef FAISS_PDX_PANORAMA_H
#define FAISS_PDX_PANORAMA_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

namespace faiss {

/// PDX vertical layout with dense compaction for progressive filtering.
///
/// Memory layout per batch of `batch_size` vectors (within one inverted list):
///
///   [Level 0: dim-major] [Level 1: dim-major] ... [Level N-1: dim-major]
///
///   Each level region: w_level × batch_size floats, stored as
///       dim0[vec0 vec1 ... vec_{B-1}]  dim1[vec0 vec1 ... vec_{B-1}]  ...
///
/// All levels use the dense vertical SIMD kernel (broadcast query dim, FMA
/// across all active candidates). After each level's pruning, survivors' data
/// for the next level is physically compacted into a dense temporary buffer
/// so the dense kernel always operates on contiguous slots — no gather/scatter.
struct PDXPanorama {
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    explicit PDXPanorama(size_t code_size, size_t n_levels, size_t batch_size)
            : code_size(code_size), n_levels(n_levels), batch_size(batch_size) {
        this->d = code_size / sizeof(float);
        this->level_width_floats = ((d + n_levels - 1) / n_levels);
        this->level_width = this->level_width_floats * sizeof(float);
    }

    /// Copy codes into vertical (dim-major) layout at a given offset.
    /// Each level's data is stored as dim_t[vec0, vec1, ..., vec_{B-1}].
    void copy_codes_to_vertical_level_layout(
            uint8_t* codes,
            size_t offset,
            size_t n_entry,
            const uint8_t* code) const {
        const float* vectors = reinterpret_cast<const float*>(code);

        for (size_t entry_idx = 0; entry_idx < n_entry; entry_idx++) {
            size_t current_pos = offset + entry_idx;
            size_t batch_no = current_pos / batch_size;
            size_t pos_in_batch = current_pos % batch_size;

            const float* src_vec = vectors + entry_idx * d;
            uint8_t* batch_base = codes + batch_no * batch_size * code_size;

            for (size_t level = 0; level < n_levels; level++) {
                const size_t start_dim = level * level_width_floats;
                const size_t end_dim =
                        std::min((level + 1) * level_width_floats, d);
                const size_t w_level = end_dim - start_dim;

                uint8_t* level_u8 =
                        batch_base + level * level_width * batch_size;
                float* level_f = reinterpret_cast<float*>(level_u8);

                for (size_t t = 0; t < w_level; t++) {
                    level_f[t * batch_size + pos_in_batch] =
                            src_vec[start_dim + t];
                }
            }
        }
    }

    template <typename C, MetricType M>
    size_t progressive_filter_batch(
            const uint8_t* codes_base,
            const float* cum_sums,
            const float* query,
            const float* query_cum_sums,
            size_t batch_no,
            size_t list_size,
            const IDSelector* sel,
            const idx_t* ids,
            bool use_sel,
            std::vector<uint32_t>& active_indices,
            std::vector<float>& exact_distances,
            float threshold,
            PanoramaStats& local_stats) const;

private:
    // ---------------------------------------------------------------
    // Dense vertical kernel: contiguous loads, broadcast query dim.
    // Processes `n` contiguous slots starting at exact_distances[0].
    // ---------------------------------------------------------------

    template <MetricType M>
    static inline void update_distances_dense_avx512(
            const float* query_level,
            const float* level_storage,
            size_t w_level,
            size_t n,
            float* exact_distances) {
        for (size_t t = 0; t < w_level; t++) {
            const float* dim_base = level_storage + t * n;

            __m512 qv;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                qv = _mm512_set1_ps(query_level[t]);
            } else {
                qv = _mm512_set1_ps(-2.0f * query_level[t]);
            }

            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m512 yv = _mm512_loadu_ps(dim_base + i);
                __m512 prev = _mm512_loadu_ps(exact_distances + i);
                prev = _mm512_fmadd_ps(qv, yv, prev);
                _mm512_storeu_ps(exact_distances + i, prev);
            }
            if (i < n) {
                __mmask16 tail_mask =
                        (__mmask16)((1u << (n - i)) - 1u);
                __m512 yv = _mm512_maskz_loadu_ps(tail_mask, dim_base + i);
                __m512 prev = _mm512_maskz_loadu_ps(
                        tail_mask, exact_distances + i);
                prev = _mm512_fmadd_ps(qv, yv, prev);
                _mm512_mask_storeu_ps(
                        exact_distances + i, tail_mask, prev);
            }
        }
    }

    // ---------------------------------------------------------------
    // Vectorized bound check on contiguous exact_distances[0..n-1]
    // and contiguous cum_sums[0..n-1]. Produces a bitmask of which
    // of the n candidates survive.
    // Returns: number of survivors written into keep_mask_out.
    //
    // Unlike the old compact_active_avx512, this operates on DENSE
    // contiguous arrays (no gather needed for distances/cumsums).
    // ---------------------------------------------------------------

    template <typename C, MetricType M>
    static inline size_t compute_keep_mask_dense(
            const float* exact_distances,
            const float* level_cum_sums,
            float query_cum_norm,
            float threshold,
            size_t n,
            uint8_t* keep_bytes) {
        const __m512 v_threshold = _mm512_set1_ps(threshold);
        const __m512 v_query_cum_norm = _mm512_set1_ps(query_cum_norm);
        const __m512 v_two = _mm512_set1_ps(2.0f);

        size_t num_kept = 0;
        size_t i = 0;

        for (; i + 16 <= n; i += 16) {
            __m512 v_dist = _mm512_loadu_ps(exact_distances + i);
            __m512 v_cum = _mm512_loadu_ps(level_cum_sums + i);

            __m512 v_lower;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                v_lower = _mm512_fmadd_ps(
                        v_cum, v_query_cum_norm, v_dist);
            } else {
                v_lower = _mm512_fnmadd_ps(
                        v_two,
                        _mm512_mul_ps(v_cum, v_query_cum_norm),
                        v_dist);
            }

            __mmask16 keep_mask;
            if constexpr (C::is_max) {
                keep_mask = _mm512_cmp_ps_mask(
                        v_lower, v_threshold, _CMP_LT_OS);
            } else {
                keep_mask = _mm512_cmp_ps_mask(
                        v_lower, v_threshold, _CMP_GT_OS);
            }

            // Store 16 bytes of 0/1 keep decisions
            for (size_t j = 0; j < 16; j++) {
                keep_bytes[i + j] = (keep_mask >> j) & 1u;
            }
            num_kept += _mm_popcnt_u32(keep_mask);
        }

        // Scalar tail
        for (; i < n; i++) {
            float dist_val = exact_distances[i];
            float cum_val = level_cum_sums[i];

            float lower_bound;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                lower_bound = dist_val + cum_val * query_cum_norm;
            } else {
                lower_bound =
                        dist_val - 2.0f * cum_val * query_cum_norm;
            }

            bool keep = C::cmp(threshold, lower_bound);
            keep_bytes[i] = keep ? 1 : 0;
            num_kept += keep ? 1 : 0;
        }

        return num_kept;
    }

};

template <typename C, MetricType M>
size_t PDXPanorama::progressive_filter_batch(
        const uint8_t* codes_base,
        const float* cum_sums,
        const float* query,
        const float* query_cum_sums,
        size_t batch_no,
        size_t list_size,
        const IDSelector* sel,
        const idx_t* ids,
        bool use_sel,
        std::vector<uint32_t>& active_indices,
        std::vector<float>& exact_distances,
        float threshold,
        PanoramaStats& local_stats) const {
    const size_t batch_start = batch_no * batch_size;
    if (batch_start >= list_size) {
        return 0;
    }
    const size_t cur_batch_size =
            std::min(batch_size, list_size - batch_start);

    if (active_indices.size() < batch_size) {
        active_indices.resize(batch_size);
    }
    if (exact_distances.size() < batch_size) {
        exact_distances.resize(batch_size);
    }

    const size_t cumsum_batch_offset = batch_no * batch_size * (n_levels + 1);
    const float* batch_cum_sums = cum_sums + cumsum_batch_offset;
    const float* level_cum_sums = batch_cum_sums + batch_size;

    const float q_norm = query_cum_sums[0] * query_cum_sums[0];

    const size_t batch_offset = batch_no * batch_size * code_size;
    const uint8_t* storage_base = codes_base + batch_offset;

    // ---- Initialize distances for all vectors in batch ----
    // active_indices here tracks the mapping from compact position
    // back to original batch position (for final heap update).
    size_t num_active = 0;
    for (size_t i = 0; i < cur_batch_size; i++) {
        const size_t global_idx = batch_start + i;
        const idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
        const bool include = !use_sel || sel->is_member(id);

        active_indices[num_active] = i;
        const float cum_sum = batch_cum_sums[i];

        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[num_active] = 0.0f;
        } else {
            exact_distances[num_active] = cum_sum * cum_sum + q_norm;
        }

        num_active += include;
    }

    if (num_active == 0) {
        return 0;
    }

    size_t total_active = num_active;
    local_stats.total_dims += total_active * n_levels;

    // Temporary buffers for compaction (reused across levels).
    // compact_level_buf: compacted vertical data for the current level.
    // cum_buf_a / cum_buf_b: ping-pong buffers for cum_sums to avoid
    //   read/write overlap during in-place compaction.
    // keep_bytes: 0/1 for each candidate after bound check.
    std::vector<float> compact_level_buf(level_width_floats * batch_size);
    std::vector<float> cum_buf_a(n_levels * batch_size);
    std::vector<float> cum_buf_b(n_levels * batch_size);
    std::vector<uint8_t> keep_bytes(batch_size);

    float* cum_src = cum_buf_a.data();
    float* cum_dst = cum_buf_b.data();

    // Gather initial cum_sums for active candidates from original storage.
    // cum_sums layout: level_cum_sums[level * batch_size + pos_in_batch]
    for (size_t i = 0; i < num_active; i++) {
        uint32_t orig = active_indices[i];
        for (size_t lv = 0; lv < n_levels; lv++) {
            cum_src[lv * num_active + i] =
                    level_cum_sums[lv * batch_size + orig];
        }
    }

    // Gather level 0's vertical data for active candidates.
    const float* level0_storage =
            reinterpret_cast<const float*>(storage_base);
    const size_t w0 = std::min(level_width_floats, d);
    for (size_t t = 0; t < w0; t++) {
        for (size_t i = 0; i < num_active; i++) {
            uint32_t orig = active_indices[i];
            compact_level_buf[t * num_active + i] =
                    level0_storage[t * batch_size + orig];
        }
    }

    // ---- Process all levels with dense kernel + compaction ----
    for (size_t level = 0; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[level + 1];
        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        // Dense vertical kernel on compacted data.
        update_distances_dense_avx512<M>(
                query_level,
                compact_level_buf.data(),
                w_level,
                num_active,
                exact_distances.data());

        // Dense bound check — contiguous arrays, no gather needed.
        size_t num_kept = compute_keep_mask_dense<C, M>(
                exact_distances.data(),
                cum_src,
                query_cum_norm,
                threshold,
                num_active,
                keep_bytes.data());

        if (num_kept == 0) {
            return 0;
        }

        // If this is the last level, compact active_indices and
        // exact_distances for the caller, then return.
        if (level == n_levels - 1) {
            size_t out = 0;
            for (size_t i = 0; i < num_active; i++) {
                if (keep_bytes[i]) {
                    active_indices[out] = active_indices[i];
                    exact_distances[out] = exact_distances[i];
                    out++;
                }
            }
            num_active = out;
            break;
        }

        // Compact everything for the next level.
        const size_t next_level = level + 1;
        const size_t next_start_dim = next_level * level_width_floats;
        const size_t w_next = std::min(level_width_floats, d - next_start_dim);
        const size_t remaining_levels = n_levels - next_level;

        // Read next level's vertical data from original storage.
        const size_t next_level_offset =
                next_level * level_width * batch_size;
        const float* next_level_storage = reinterpret_cast<const float*>(
                storage_base + next_level_offset);

        size_t out = 0;
        for (size_t i = 0; i < num_active; i++) {
            if (!keep_bytes[i]) {
                continue;
            }

            exact_distances[out] = exact_distances[i];
            active_indices[out] = active_indices[i];

            // Compact cum_sums for remaining levels into dst buffer.
            // Source: cum_src[(lv+1) * num_active + i] (skip consumed level).
            // Dest:   cum_dst[lv * num_kept + out].
            for (size_t lv = 0; lv < remaining_levels; lv++) {
                cum_dst[lv * num_kept + out] =
                        cum_src[(lv + 1) * num_active + i];
            }

            // Compact next level's vertical data from original storage.
            // Source: next_level_storage[t * batch_size + orig_batch_pos]
            // Dest:   compact_level_buf[t * num_kept + out]
            uint32_t orig = active_indices[i];
            for (size_t t = 0; t < w_next; t++) {
                compact_level_buf[t * num_kept + out] =
                        next_level_storage[t * batch_size + orig];
            }

            out++;
        }

        num_active = num_kept;

        // Swap ping-pong buffers.
        std::swap(cum_src, cum_dst);
    }

    return num_active;
}

} // namespace faiss

#endif