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

/// PDX hybrid layout for progressive filtering.
///
/// Memory layout per batch of `batch_size` vectors:
///
///   [Vertical levels 0..K-1: dim-major]
///       For each level: w_level × batch_size floats
///       dim0[vec0 vec1 ... vec_{B-1}]  dim1[vec0 ... vec_{B-1}]  ...
///
///   [Horizontal levels K..N-1: vector-major]
///       batch_size × horiz_dims floats
///       vec0[dim_{K*w} dim_{K*w+1} ... dim_{d-1}]
///       vec1[dim_{K*w} dim_{K*w+1} ... dim_{d-1}]
///       ...
///
/// The first K levels use the dense vertical SIMD kernel (broadcast query dim,
/// FMA across all batch_size candidates). After K levels of pruning, the
/// remaining levels use Panorama-style fvec_inner_product per survivor on
/// contiguous per-vector storage — zero wasted work on dead candidates.
struct PDXPanorama {
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    // Number of levels stored in vertical (dim-major) layout.
    size_t n_vertical_levels = 0;
    // Total number of floats across all vertical levels.
    size_t vert_total_floats = 0;
    // Number of remaining floats per vector (horizontal region).
    size_t horiz_dims = 0;
    // Byte offset where horizontal region starts within a batch.
    size_t vert_region_bytes = 0;

    explicit PDXPanorama(
            size_t code_size,
            size_t n_levels,
            size_t batch_size,
            size_t n_vertical_levels = 2)
            : code_size(code_size),
              n_levels(n_levels),
              batch_size(batch_size),
              n_vertical_levels(std::min(n_vertical_levels, n_levels)) {
        this->d = code_size / sizeof(float);
        this->level_width_floats = ((d + n_levels - 1) / n_levels);
        this->level_width = this->level_width_floats * sizeof(float);
        this->vert_total_floats = std::min(
                this->n_vertical_levels * level_width_floats, d);
        this->horiz_dims = d - vert_total_floats;
        this->vert_region_bytes =
                this->n_vertical_levels * level_width * batch_size;
    }

    /// Copy codes into hybrid layout at a given offset.
    /// First n_vertical_levels: dim-major (vertical).
    /// Remaining levels: vector-major (horizontal).
    void copy_codes_to_hybrid_layout(
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

            // Vertical levels: dim-major.
            for (size_t level = 0; level < n_vertical_levels; level++) {
                const size_t start_dim = level * level_width_floats;
                const size_t end_dim =
                        std::min((level + 1) * level_width_floats, d);
                const size_t w_level = end_dim - start_dim;

                float* level_f = reinterpret_cast<float*>(
                        batch_base + level * level_width * batch_size);

                for (size_t t = 0; t < w_level; t++) {
                    level_f[t * batch_size + pos_in_batch] =
                            src_vec[start_dim + t];
                }
            }

            // Horizontal levels: vector-major.
            if (horiz_dims > 0) {
                float* horiz_base = reinterpret_cast<float*>(
                        batch_base + vert_region_bytes);
                float* dest = horiz_base + pos_in_batch * horiz_dims;
                memcpy(dest,
                       src_vec + vert_total_floats,
                       horiz_dims * sizeof(float));
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
    // Processes all `n` slots (stride between dims = `stride`).
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
    // exact_distances and cum_sums indexed by batch position via
    // active_indices.
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
    // exact_distances indexed by batch position throughout.
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
    // Vertical levels 0..K-1: Dense vertical SIMD kernel on all
    // batch_size slots + gather-based bound check.
    // ============================================================
    for (size_t level = 0; level < n_vertical_levels; level++) {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[level + 1];
        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        const float* level_storage = reinterpret_cast<const float*>(
                storage_base + level * level_width * batch_size);

        update_distances_dense_avx512<M>(
                query_level,
                level_storage,
                w_level,
                batch_size,
                batch_size,
                exact_distances.data());

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
    }

    // ============================================================
    // Horizontal levels K..N-1: Panorama-style fvec_inner_product
    // per survivor on vector-major storage.
    // ============================================================
    if (n_vertical_levels < n_levels) {
        const float* horiz_base = reinterpret_cast<const float*>(
                storage_base + vert_region_bytes);

        for (size_t level = n_vertical_levels; level < n_levels; level++) {
            local_stats.total_dims_scanned += num_active;

            const float query_cum_norm = query_cum_sums[level + 1];
            const size_t start_dim = level * level_width_floats;
            const size_t actual_level_width =
                    std::min(level_width_floats, d - start_dim);
            const float* query_level = query + start_dim;

            // Offset within the horizontal region for this level's dims.
            const size_t horiz_level_offset = start_dim - vert_total_floats;

            size_t next_active = 0;
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];

                const float* yj =
                        horiz_base + idx * horiz_dims + horiz_level_offset;

                float dot_product = fvec_inner_product(
                        query_level, yj, actual_level_width);

                if constexpr (M == METRIC_INNER_PRODUCT) {
                    exact_distances[idx] += dot_product;
                } else {
                    exact_distances[idx] -= 2.0f * dot_product;
                }

                float cum_sum = level_cum_sums[idx];
                float cauchy_schwarz_bound;
                if constexpr (M == METRIC_INNER_PRODUCT) {
                    cauchy_schwarz_bound = -cum_sum * query_cum_norm;
                } else {
                    cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
                }

                float lower_bound =
                        exact_distances[idx] - cauchy_schwarz_bound;

                active_indices[next_active] = idx;
                next_active += C::cmp(threshold, lower_bound) ? 1 : 0;
            }

            num_active = next_active;
            level_cum_sums += batch_size;

            if (num_active == 0) {
                return 0;
            }
        }
    }

    return num_active;
}

} // namespace faiss

#endif