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

/// PDX all-vertical layout with bitset-based code compression.
///
/// Memory layout per batch of `batch_size` vectors (same as original PDX):
///
///   [Level 0: dim-major] [Level 1: dim-major] ... [Level N-1: dim-major]
///
///   Each level region: w_level × batch_size floats, stored as
///       dim0[vec0 vec1 ... vec_{B-1}]  dim1[vec0 vec1 ... vec_{B-1}]  ...
///
/// Processing strategy (inspired by IVFPQ Panorama scanner):
///
/// Level 0: Dense vertical kernel on all batch_size slots (all alive).
///   After filtering, build a byte bitset marking survivors.
///
/// Levels 1+: Use the bitset to compress the next level's vertical data,
///   exact_distances, and cum_sums into dense buffers (16 floats at a time
///   via _mm512_mask_compressstoreu_ps). Then run the dense kernel on
///   only num_active slots — zero wasted FMAs. After filtering, rebuild
///   the bitset for the next level.
///
/// This combines the SIMD throughput of the vertical layout with the
/// zero-waste property of per-survivor processing.
struct PDXPanorama {
    size_t d = 0;
    size_t code_size = 0;
    size_t n_levels = 0;
    size_t level_width = 0;
    size_t level_width_floats = 0;
    size_t batch_size = 0;

    static constexpr size_t kMaxBatchSize = 128;

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
    // Processes `n` slots with stride between dims = `stride`.
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
    // Gather-based bound check + compressstore compaction.
    // exact_distances and cum_sums indexed by batch position via
    // active_indices. Used for level 0 where data is at batch positions.
    // ---------------------------------------------------------------

    template <typename C, MetricType M>
    static inline size_t compact_active_gather(
            uint32_t* active_indices,
            const float* exact_distances,
            const float* level_cum_sums,
            float query_cum_norm,
            float threshold,
            size_t num_active) {
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
    // Dense bound check + compaction on contiguous arrays.
    // Used for levels 1+ after bitset compression has made data dense.
    // Returns new num_active; compacts exact_distances, active_indices,
    // and writes bitset for the next compression pass.
    // ---------------------------------------------------------------

    template <typename C, MetricType M>
    static inline size_t filter_dense(
            float* exact_distances,
            uint32_t* active_indices,
            const float* cum_sums_dense,
            float query_cum_norm,
            float threshold,
            size_t num_active,
            uint8_t* bitset,
            size_t batch_size) {
        const __m512 v_threshold = _mm512_set1_ps(threshold);
        const __m512 v_query_cum_norm = _mm512_set1_ps(query_cum_norm);
        const __m512 v_two = _mm512_set1_ps(2.0f);

        size_t next_active = 0;
        size_t i = 0;

        // Zero the bitset (batch_size is always a multiple of 64).
        memset(bitset, 0, batch_size);

        for (; i + 16 <= num_active; i += 16) {
            __m512 v_dist = _mm512_loadu_ps(exact_distances + i);
            __m512 v_cum = _mm512_loadu_ps(cum_sums_dense + i);

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

            // Compact exact_distances.
            _mm512_mask_compressstoreu_ps(
                    exact_distances + next_active, keep_mask, v_dist);

            // Compact active_indices.
            __m512i v_idx = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(active_indices + i));
            _mm512_mask_compressstoreu_epi32(
                    active_indices + next_active, keep_mask, v_idx);

            // Set bitset for survivors (used for next level's compression).
            // Load original batch indices for survivors to mark in bitset.
            alignas(64) uint32_t kept_indices[16];
            __m512i compressed_idx = _mm512_mask_compress_epi32(
                    _mm512_setzero_si512(), keep_mask, v_idx);
            _mm512_storeu_si512(kept_indices, compressed_idx);
            size_t n_kept = _mm_popcnt_u32(keep_mask);
            for (size_t j = 0; j < n_kept; j++) {
                bitset[kept_indices[j]] = 1;
            }

            next_active += n_kept;
        }

        // Scalar tail.
        for (; i < num_active; i++) {
            float dist_val = exact_distances[i];
            float cum_val = cum_sums_dense[i];

            float lower_bound;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                lower_bound = dist_val + cum_val * query_cum_norm;
            } else {
                lower_bound =
                        dist_val - 2.0f * cum_val * query_cum_norm;
            }

            if (C::cmp(threshold, lower_bound)) {
                exact_distances[next_active] = dist_val;
                active_indices[next_active] = active_indices[i];
                bitset[active_indices[i]] = 1;
                next_active++;
            }
        }

        return next_active;
    }

    // ---------------------------------------------------------------
    // Bitset-based float compression: compress vertical data for one
    // level from batch_size → num_active using the bitset.
    // Processes 16 floats at a time per dimension.
    // ---------------------------------------------------------------

    static inline void compress_vertical_level(
            const float* src_level,
            float* dst_level,
            const uint8_t* bitset,
            size_t w_level,
            size_t batch_size,
            size_t num_active) {
        // Build mask from bitset, 16 entries at a time, and compress
        // each dimension's data.
        for (size_t t = 0; t < w_level; t++) {
            const float* src_dim = src_level + t * batch_size;
            float* dst_dim = dst_level + t * num_active;
            size_t out = 0;

            size_t pos = 0;
            for (; pos + 16 <= batch_size; pos += 16) {
                // Load 16 bytes of bitset, compare to zero → 16-bit mask.
                __m128i bs = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(bitset + pos));
                __mmask16 mask = _mm_cmpneq_epi8_mask(
                        bs, _mm_setzero_si128());

                __m512 vals = _mm512_loadu_ps(src_dim + pos);
                _mm512_mask_compressstoreu_ps(
                        dst_dim + out, mask, vals);
                out += _mm_popcnt_u32(mask);
            }
            // Scalar tail.
            for (; pos < batch_size; pos++) {
                if (bitset[pos]) {
                    dst_dim[out++] = src_dim[pos];
                }
            }
        }
    }

    // Compress a single flat array (exact_distances or cum_sums) using bitset.
    static inline size_t compress_flat_array(
            const float* src,
            float* dst,
            const uint8_t* bitset,
            size_t batch_size) {
        size_t out = 0;
        size_t pos = 0;
        for (; pos + 16 <= batch_size; pos += 16) {
            __m128i bs = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(bitset + pos));
            __mmask16 mask = _mm_cmpneq_epi8_mask(
                    bs, _mm_setzero_si128());

            __m512 vals = _mm512_loadu_ps(src + pos);
            _mm512_mask_compressstoreu_ps(dst + out, mask, vals);
            out += _mm_popcnt_u32(mask);
        }
        for (; pos < batch_size; pos++) {
            if (bitset[pos]) {
                dst[out++] = src[pos];
            }
        }
        return out;
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
    // exact_distances indexed by batch position for level 0.
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
    // Level 0: Dense vertical kernel on all batch_size slots.
    // ============================================================
    {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[1];
        const float* level0_storage =
                reinterpret_cast<const float*>(storage_base);
        const size_t w0 = std::min(level_width_floats, d);

        update_distances_dense_avx512<M>(
                query,
                level0_storage,
                w0,
                batch_size,
                batch_size,
                exact_distances.data());

        // Gather-based filter (data still at batch positions).
        num_active = compact_active_gather<C, M>(
                active_indices.data(),
                exact_distances.data(),
                level_cum_sums,
                query_cum_norm,
                threshold,
                num_active);

        level_cum_sums += batch_size;
        if (num_active == 0 || n_levels == 1) {
            return num_active;
        }
    }

    // ============================================================
    // Build bitset from level 0 survivors + compress for level 1.
    // ============================================================
    // Scratch buffers for compacted data. Allocated once, reused.
    // We need: level data (w_level * num_active), distances (num_active),
    // cum_sums for remaining levels (num_active each), bitset (batch_size).
    alignas(64) uint8_t byteset[kMaxBatchSize];
    memset(byteset, 0, kMaxBatchSize);
    for (size_t i = 0; i < num_active; i++) {
        byteset[active_indices[i]] = 1;
    }

    // Compress exact_distances from batch positions to dense [0..num_active-1].
    // We need a temp buffer to avoid read/write overlap.
    alignas(64) float compact_dist[kMaxBatchSize];
    compress_flat_array(
            exact_distances.data(), compact_dist,
            byteset, batch_size);
    memcpy(exact_distances.data(), compact_dist,
           num_active * sizeof(float));

    // Scratch for compressed level data and cum_sums.
    std::vector<float> compact_level(level_width_floats * batch_size);
    alignas(64) float compact_cum[kMaxBatchSize];

    // ============================================================
    // Levels 1+: Compress → dense kernel → dense filter → repeat.
    // ============================================================
    for (size_t level = 1; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;

        const float query_cum_norm = query_cum_sums[level + 1];
        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        const float* level_storage = reinterpret_cast<const float*>(
                storage_base + level * level_width * batch_size);

        // Compress this level's vertical data using bitset.
        compress_vertical_level(
                level_storage, compact_level.data(),
                byteset, w_level, batch_size, num_active);

        // Compress this level's cum_sums.
        compress_flat_array(
                level_cum_sums, compact_cum,
                byteset, batch_size);

        // Dense vertical kernel on compacted data (stride = num_active).
        update_distances_dense_avx512<M>(
                query_level,
                compact_level.data(),
                w_level,
                num_active,
                num_active,
                exact_distances.data());

        // Dense filter: contiguous arrays, no gathers.
        // Also rebuilds bitset for next level.
        num_active = filter_dense<C, M>(
                exact_distances.data(),
                active_indices.data(),
                compact_cum,
                query_cum_norm,
                threshold,
                num_active,
                byteset,
                batch_size);

        level_cum_sums += batch_size;
        if (num_active == 0) {
            return 0;
        }
    }

    // Scatter exact_distances back to batch positions for caller.
    // Use compact_dist as temp to avoid overwrites.
    memcpy(compact_dist, exact_distances.data(),
           num_active * sizeof(float));
    for (size_t i = 0; i < num_active; i++) {
        exact_distances[active_indices[i]] = compact_dist[i];
    }

    return num_active;
}

} // namespace faiss

#endif