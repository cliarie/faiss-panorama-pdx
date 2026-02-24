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

/// PDX vertical layout with two-phase progressive filtering.
///
/// Memory layout per batch of `batch_size` vectors (within one inverted list):
///
///   [Level 0: dim-major] [Level 1: dim-major] ... [Level N-1: dim-major]
///
///   Each level region: w_level × batch_size floats, stored as
///       dim0[vec0 vec1 ... vec_{B-1}]  dim1[vec0 vec1 ... vec_{B-1}]  ...
///
/// Phase 1 (dense, no compaction): While num_active > kCompactThreshold,
/// run the dense vertical SIMD kernel on all batch_size slots and use
/// gather-based bound checking to prune the active_indices list.
/// No data is moved — wasted FMAs on dead slots are cheaper than compaction.
///
/// Phase 2 (compacted dense): Once num_active drops below kCompactThreshold,
/// do a one-time compaction of survivors' remaining data into dense buffers,
/// then continue with the dense kernel on the smaller compacted set.
struct PDXPanorama {
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    static constexpr size_t kCompactThreshold = 96;

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
    // Stride is the distance between consecutive vectors for one dim.
    // ---------------------------------------------------------------

    template <MetricType M>
    static inline void update_distances_dense_avx512(
            const float* query_level,
            const float* level_storage,
            size_t w_level,
            size_t n,
            size_t stride,
            float* exact_distances) {
        for (size_t t = 0; t < w_level; t++) {
            const float* dim_base = level_storage + t * stride;

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
    // Vectorized bound check + compressstore compaction (gather-based).
    // Used in Phase 1 where exact_distances and cum_sums are at
    // original batch positions (indexed by active_indices).
    // ---------------------------------------------------------------

    template <typename C, MetricType M>
    static inline size_t compact_active_avx512(
            uint32_t* active_indices,
            const float* exact_distances,
            const float* level_cum_sums,
            float query_cum_norm,
            float threshold,
            size_t num_active,
            size_t batch_size) {
        const __m512 v_threshold = _mm512_set1_ps(threshold);
        const __m512 v_query_cum_norm = _mm512_set1_ps(query_cum_norm);
        const __m512 v_two = _mm512_set1_ps(2.0f);

        size_t next_active = 0;
        size_t i = 0;

        for (; i + 16 <= num_active; i += 16) {
            __m512i vidx = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(active_indices + i));

            __m512 v_dist = _mm512_i32gather_ps(vidx, exact_distances, 4);
            __m512 v_cum = _mm512_i32gather_ps(vidx, level_cum_sums, 4);

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

            _mm512_mask_compressstoreu_epi32(
                    active_indices + next_active, keep_mask, vidx);
            next_active += _mm_popcnt_u32(keep_mask);
        }

        if (i < num_active) {
            __mmask16 tail_mask =
                    (__mmask16)((1u << (num_active - i)) - 1u);
            __m512i vidx = _mm512_maskz_loadu_epi32(
                    tail_mask,
                    reinterpret_cast<const void*>(active_indices + i));

            __m512 v_dist = _mm512_mask_i32gather_ps(
                    _mm512_setzero_ps(), tail_mask, vidx, exact_distances, 4);
            __m512 v_cum = _mm512_mask_i32gather_ps(
                    _mm512_setzero_ps(), tail_mask, vidx, level_cum_sums, 4);

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
            keep_mask &= tail_mask;

            _mm512_mask_compressstoreu_epi32(
                    active_indices + next_active, keep_mask, vidx);
            next_active += _mm_popcnt_u32(keep_mask);
        }

        return next_active;
    }

    // ---------------------------------------------------------------
    // Dense bound check on contiguous arrays (used in Phase 2).
    // Returns number of survivors; writes keep_bytes[0..n-1].
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

            for (size_t j = 0; j < 16; j++) {
                keep_bytes[i + j] = (keep_mask >> j) & 1u;
            }
            num_kept += _mm_popcnt_u32(keep_mask);
        }

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

    // ---- Initialize active set with ID-filtered vectors ----
    // In Phase 1, exact_distances is indexed by batch position.
    size_t num_active = 0;
    for (size_t i = 0; i < cur_batch_size; i++) {
        const size_t global_idx = batch_start + i;
        const idx_t id = (ids == nullptr) ? global_idx : ids[global_idx];
        const bool include = !use_sel || sel->is_member(id);

        active_indices[num_active] = i;
        const float cum_sum = batch_cum_sums[i];

        if constexpr (M == METRIC_INNER_PRODUCT) {
            exact_distances[i] = 0.0f;
        } else {
            exact_distances[i] = cum_sum * cum_sum + q_norm;
        }

        num_active += include;
    }

    for (size_t i = cur_batch_size; i < batch_size; i++) {
        exact_distances[i] = 0.0f;
    }

    if (num_active == 0) {
        return 0;
    }

    size_t total_active = num_active;
    local_stats.total_dims += total_active * n_levels;

    // ============================================================
    // Phase 1: Dense vertical kernel on full batch, gather-based
    // bound check. No data compaction — just maintain active_indices.
    // ============================================================
    size_t level = 0;
    for (; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[level + 1];
        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        const float* level_storage = reinterpret_cast<const float*>(
                storage_base + level * level_width * batch_size);

        // Dense kernel on all batch_size slots (stride = batch_size).
        update_distances_dense_avx512<M>(
                query_level,
                level_storage,
                w_level,
                batch_size,
                batch_size,
                exact_distances.data());

        // Gather-based bound check + compressstore compaction of indices.
        num_active = compact_active_avx512<C, M>(
                active_indices.data(),
                exact_distances.data(),
                level_cum_sums,
                query_cum_norm,
                threshold,
                num_active,
                batch_size);

        level_cum_sums += batch_size;
        if (num_active == 0) {
            return 0;
        }

        // Check if we should switch to Phase 2.
        if (num_active <= kCompactThreshold && level + 1 < n_levels) {
            level++;
            break;
        }
    }

    // If all levels done in Phase 1, return with batch-position indexing.
    if (level >= n_levels) {
        return num_active;
    }

    // ============================================================
    // Phase 2: One-time compaction, then dense kernel on compact set.
    // Compact survivors' exact_distances, cum_sums, and vertical data
    // for remaining levels into dense buffers.
    // ============================================================

    // Allocate compact buffers.
    std::vector<float> compact_level_buf(level_width_floats * num_active);
    std::vector<float> cum_buf_a(
            (n_levels - level) * num_active);
    std::vector<float> cum_buf_b(
            (n_levels - level) * num_active);
    std::vector<uint8_t> keep_bytes(num_active);

    float* cum_src = cum_buf_a.data();
    float* cum_dst = cum_buf_b.data();

    const size_t remaining_levels = n_levels - level;

    // Compact exact_distances, cum_sums, and current level's data.
    const float* cur_level_storage = reinterpret_cast<const float*>(
            storage_base + level * level_width * batch_size);
    const size_t cur_start_dim = level * level_width_floats;
    const size_t w_cur = std::min(level_width_floats, d - cur_start_dim);

    for (size_t i = 0; i < num_active; i++) {
        uint32_t orig = active_indices[i];

        // Compact exact_distances from batch position to dense position.
        exact_distances[i] = exact_distances[orig];

        // Compact cum_sums for remaining levels.
        for (size_t lv = 0; lv < remaining_levels; lv++) {
            cum_src[lv * num_active + i] =
                    level_cum_sums[lv * batch_size + orig];
        }

        // Compact current level's vertical data.
        for (size_t t = 0; t < w_cur; t++) {
            compact_level_buf[t * num_active + i] =
                    cur_level_storage[t * batch_size + orig];
        }
    }

    // Process remaining levels in Phase 2.
    for (size_t lv_offset = 0; level < n_levels; level++, lv_offset++) {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[level + 1];
        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        // Dense vertical kernel on compacted data (stride = num_active).
        update_distances_dense_avx512<M>(
                query_level,
                compact_level_buf.data(),
                w_level,
                num_active,
                num_active,
                exact_distances.data());

        // Dense bound check — contiguous arrays, no gather.
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

        // Last level: compact active_indices + exact_distances for caller.
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

        // Compact for next level.
        const size_t next_level = level + 1;
        const size_t next_start_dim = next_level * level_width_floats;
        const size_t w_next =
                std::min(level_width_floats, d - next_start_dim);
        const size_t next_remaining = n_levels - next_level;

        const float* next_level_storage = reinterpret_cast<const float*>(
                storage_base + next_level * level_width * batch_size);

        size_t out = 0;
        for (size_t i = 0; i < num_active; i++) {
            if (!keep_bytes[i]) {
                continue;
            }

            exact_distances[out] = exact_distances[i];
            active_indices[out] = active_indices[i];

            // Compact cum_sums: skip consumed level.
            for (size_t lv = 0; lv < next_remaining; lv++) {
                cum_dst[lv * num_kept + out] =
                        cum_src[(lv + 1) * num_active + i];
            }

            // Compact next level's vertical data from original storage.
            uint32_t orig = active_indices[i];
            for (size_t t = 0; t < w_next; t++) {
                compact_level_buf[t * num_kept + out] =
                        next_level_storage[t * batch_size + orig];
            }

            out++;
        }

        num_active = num_kept;
        std::swap(cum_src, cum_dst);
    }

    // Phase 2 returns with dense-position indexing for exact_distances.
    // Scatter back to batch positions for caller compatibility.
    // Use a temp buffer to avoid overwrites.
    std::vector<float> tmp_dist(num_active);
    for (size_t i = 0; i < num_active; i++) {
        tmp_dist[i] = exact_distances[i];
    }
    for (size_t i = 0; i < num_active; i++) {
        exact_distances[active_indices[i]] = tmp_dist[i];
    }

    return num_active;
}

} // namespace faiss

#endif