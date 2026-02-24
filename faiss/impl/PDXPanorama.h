#ifndef FAISS_PDX_PANORAMA_H
#define FAISS_PDX_PANORAMA_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <immintrin.h>

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

namespace faiss {

/// Hybrid PDX+Panorama layout for progressive filtering.
///
/// Memory layout per batch of `batch_size` vectors (within one inverted list):
///
///   [Level 0: vertical (dim-major)]  [Levels 1..N-1: horizontal (vector-major)]
///
///   Level 0 region:  w0 × batch_size floats, stored as
///       dim0[vec0 vec1 ... vec_{B-1}]  dim1[vec0 vec1 ... vec_{B-1}]  ...
///
///   Levels 1+ region: batch_size × remaining_dims floats, stored as
///       vec0[dim_{w0} dim_{w0+1} ... dim_{d-1}]
///       vec1[dim_{w0} dim_{w0+1} ... dim_{d-1}]  ...
///
/// Level 0 is always computed for all candidates (no fragmentation), enabling
/// dense contiguous SIMD (broadcast query dim, FMA across all batch slots).
/// After level-0 pruning, surviving candidates' remaining dimensions are
/// contiguous in the horizontal region, enabling fast sequential loads via
/// fvec_inner_product — no gather/scatter needed.
struct PDXPanorama {
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    /// Number of floats in the horizontal tail per vector (d - level_width_floats).
    size_t horiz_dims = 0;
    /// Byte size of level-0 vertical region per batch.
    size_t level0_region_bytes = 0;

    explicit PDXPanorama(size_t code_size, size_t n_levels, size_t batch_size)
            : code_size(code_size), n_levels(n_levels), batch_size(batch_size) {
        this->d = code_size / sizeof(float);
        this->level_width_floats = ((d + n_levels - 1) / n_levels);
        this->level_width = this->level_width_floats * sizeof(float);
        this->horiz_dims = d - level_width_floats;
        this->level0_region_bytes = level_width * batch_size;
    }

    /// Copy codes into hybrid layout at a given offset in the list.
    /// Level 0: vertical (dim-major within batch).
    /// Levels 1+: horizontal (vector-major, contiguous remaining dims).
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

            // Level 0: vertical (dim-major).
            // Layout: dim_t[vec0, vec1, ..., vec_{B-1}] for t in [0, w0).
            {
                const size_t w0 =
                        std::min(level_width_floats, d);
                float* level0_f = reinterpret_cast<float*>(batch_base);

                for (size_t t = 0; t < w0; t++) {
                    level0_f[t * batch_size + pos_in_batch] = src_vec[t];
                }
            }

            // Levels 1+: horizontal (vector-major).
            // Layout: vec_i[dim_{w0}, dim_{w0+1}, ..., dim_{d-1}].
            if (horiz_dims > 0) {
                float* horiz_base = reinterpret_cast<float*>(
                        batch_base + level0_region_bytes);
                float* dest = horiz_base + pos_in_batch * horiz_dims;
                const float* src = src_vec + level_width_floats;
                memcpy(dest, src, horiz_dims * sizeof(float));
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
    // Used for level 0 where all batch_size slots are processed.
    // ---------------------------------------------------------------

    template <MetricType M>
    static inline void update_distances_dense_avx512(
            const float* query_level,
            const float* level_storage,
            size_t w_level,
            size_t batch_size,
            float* exact_distances) {
        // For each dimension in this level, broadcast query value and
        // FMA across all batch_size candidates using contiguous loads.
        for (size_t t = 0; t < w_level; t++) {
            const float* dim_base = level_storage + t * batch_size;

            // For IP: exact_dist += q * y  →  fmadd(q, y, prev)
            // For L2: exact_dist -= 2*q*y  →  fmadd(-2*q, y, prev)
            __m512 qv;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                qv = _mm512_set1_ps(query_level[t]);
            } else {
                qv = _mm512_set1_ps(-2.0f * query_level[t]);
            }

            size_t i = 0;
            for (; i + 16 <= batch_size; i += 16) {
                __m512 yv = _mm512_loadu_ps(dim_base + i);
                __m512 prev = _mm512_loadu_ps(exact_distances + i);
                prev = _mm512_fmadd_ps(qv, yv, prev);
                _mm512_storeu_ps(exact_distances + i, prev);
            }
            // Masked tail (batch_size % 16 != 0)
            if (i < batch_size) {
                __mmask16 tail_mask =
                        (__mmask16)((1u << (batch_size - i)) - 1u);
                __m512 yv = _mm512_maskz_loadu_ps(tail_mask, dim_base + i);
                __m512 prev = _mm512_maskz_loadu_ps(tail_mask, exact_distances + i);
                prev = _mm512_fmadd_ps(qv, yv, prev);
                _mm512_mask_storeu_ps(exact_distances + i, tail_mask, prev);
            }
        }
    }

    // ---------------------------------------------------------------
    // Vectorized bound check + compressstore compaction.
    // Computes lower bounds for 16 candidates at a time, generates a
    // keep-mask, and uses _mm512_mask_compressstoreu_epi32 to write
    // surviving indices contiguously.
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

            // Gather exact_distances[idx] and cum_sums[idx]
            __m512 v_dist = _mm512_i32gather_ps(vidx, exact_distances, 4);
            __m512 v_cum = _mm512_i32gather_ps(vidx, level_cum_sums, 4);

            // Compute lower_bound = exact_dist - cauchy_schwarz_bound
            __m512 v_lower;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                // bound = -cum * query_cum_norm
                // lower = dist - bound = dist + cum * query_cum_norm
                v_lower = _mm512_fmadd_ps(v_cum, v_query_cum_norm, v_dist);
            } else {
                // bound = 2 * cum * query_cum_norm
                // lower = dist - 2 * cum * query_cum_norm
                v_lower = _mm512_fnmadd_ps(
                        v_two, _mm512_mul_ps(v_cum, v_query_cum_norm), v_dist);
            }

            // Generate keep mask: C::cmp(threshold, lower_bound)
            // For CMax (L2): keep if threshold > lower_bound (LT)
            // For CMin (IP): keep if threshold < lower_bound (GT)
            __mmask16 keep_mask;
            if constexpr (C::is_max) {
                // CMax: threshold > lower_bound => lower_bound < threshold
                keep_mask =
                        _mm512_cmp_ps_mask(v_lower, v_threshold, _CMP_LT_OS);
            } else {
                // CMin: threshold < lower_bound => lower_bound > threshold
                keep_mask =
                        _mm512_cmp_ps_mask(v_lower, v_threshold, _CMP_GT_OS);
            }

            // Compressstore: write surviving indices contiguously
            _mm512_mask_compressstoreu_epi32(
                    active_indices + next_active, keep_mask, vidx);
            next_active += _mm_popcnt_u32(keep_mask);
        }

        // Masked tail: process remaining < 16 indices with masked ops
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
                v_lower = _mm512_fmadd_ps(v_cum, v_query_cum_norm, v_dist);
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

    // ---- Level 0: vertical (dim-major) dense SIMD kernel ----
    {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[1];
        const float* level0_storage =
                reinterpret_cast<const float*>(storage_base);
        const size_t w0 = std::min(level_width_floats, d);
        const float* query_level0 = query;

        update_distances_dense_avx512<M>(
                query_level0,
                level0_storage,
                w0,
                batch_size,
                exact_distances.data());

        // Vectorized bound check + compressstore compaction after level 0.
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

    // ---- Levels 1+: horizontal (vector-major) Panorama-style loop ----
    // Each survivor's remaining dimensions (d - level_width_floats) are
    // stored contiguously, enabling fast fvec_inner_product.
    if (n_levels > 1 && horiz_dims > 0) {
        const float* horiz_base = reinterpret_cast<const float*>(
                storage_base + level0_region_bytes);

        for (size_t level = 1; level < n_levels; level++) {
            local_stats.total_dims_scanned += num_active;

            const float query_cum_norm = query_cum_sums[level + 1];

            const size_t start_dim = level * level_width_floats;
            const size_t w_level =
                    std::min(level_width_floats, d - start_dim);
            // Offset into the horizontal region for this level's dims.
            const size_t horiz_level_offset =
                    start_dim - level_width_floats;
            const float* query_level = query + start_dim;

            size_t next_active = 0;
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];

                const float* yj =
                        horiz_base + idx * horiz_dims + horiz_level_offset;

                float dot_product =
                        fvec_inner_product(query_level, yj, w_level);

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
                break;
            }
        }
    }

    return num_active;
}

} // namespace faiss

#endif