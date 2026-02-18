#ifndef FAISS_PDX_PANORAMA_H
#define FAISS_PDX_PANORAMA_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/utils/distances.h>

namespace faiss {

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
    template <MetricType M>
    static inline void update_exact_distances_vertical_scalar(
            const float* query_level,
            const float* level_storage,
            size_t w_level,
            const uint32_t* active_indices,
            size_t num_active,
            size_t batch_size,
            float* exact_distances) {
        for (size_t i = 0; i < num_active; i++) {
            const uint32_t idx = active_indices[i];

            float dot_product = 0.0f;
            for (size_t t = 0; t < w_level; t++) {
                dot_product += query_level[t] * level_storage[t * batch_size + idx];
            }

            if constexpr (M == METRIC_INNER_PRODUCT) {
                exact_distances[idx] += dot_product;
            } else {
                exact_distances[idx] -= 2.0f * dot_product;
            }
        }
    }

#ifdef __AVX512F__
    template <MetricType M>
    static inline void update_exact_distances_vertical_avx512(
            const float* query_level,
            const float* level_storage,
            size_t w_level,
            const uint32_t* active_indices,
            size_t num_active,
            size_t batch_size,
            float* exact_distances) {
        (void)batch_size;
        size_t i = 0;
        for (; i + 16 <= num_active; i += 16) {
            __m512 acc = _mm512_setzero_ps();
            __m512i vidx = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(active_indices + i));

            for (size_t t = 0; t < w_level; t++) {
                __m512 qv = _mm512_set1_ps(query_level[t]);
                const float* base = level_storage + t * batch_size;
                __m512 xv = _mm512_i32gather_ps(vidx, base, 4);
                acc = _mm512_fmadd_ps(qv, xv, acc);
            }

            __m512 prev = _mm512_i32gather_ps(vidx, exact_distances, 4);
            __m512 updated;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                updated = _mm512_add_ps(prev, acc);
            } else {
                updated = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), acc, prev);
            }
            _mm512_i32scatter_ps(exact_distances, vidx, updated, 4);
        }

        if (i < num_active) {
            update_exact_distances_vertical_scalar<M>(
                    query_level,
                    level_storage,
                    w_level,
                    active_indices + i,
                    num_active - i,
                    batch_size,
                    exact_distances);
        }
    }
#endif
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
    for (size_t level = 0; level < n_levels; level++) {
        local_stats.total_dims_scanned += num_active;
        local_stats.total_dims += total_active;

        const float query_cum_norm = query_cum_sums[level + 1];

        const size_t level_offset = level * level_width * batch_size;
        const float* level_storage = reinterpret_cast<const float*>(
                storage_base + level_offset);

        const size_t start_dim = level * level_width_floats;
        const size_t w_level = std::min(level_width_floats, d - start_dim);
        const float* query_level = query + start_dim;

        // Vertical kernel: update exact_distances[idx] for all active indices.
#ifdef __AVX512F__
        update_exact_distances_vertical_avx512<M>(
                query_level,
                level_storage,
                w_level,
                active_indices.data(),
                num_active,
                batch_size,
                exact_distances.data());
#else
        update_exact_distances_vertical_scalar<M>(
                query_level,
                level_storage,
                w_level,
                active_indices.data(),
                num_active,
                batch_size,
                exact_distances.data());
#endif

        size_t next_active = 0;
        for (size_t i = 0; i < num_active; i++) {
            const uint32_t idx = active_indices[i];
            const float cum_sum = level_cum_sums[idx];
            float cauchy_schwarz_bound;
            if constexpr (M == METRIC_INNER_PRODUCT) {
                cauchy_schwarz_bound = -cum_sum * query_cum_norm;
            } else {
                cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
            }

            const float lower_bound =
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

    return num_active;
}

} // namespace faiss

#endif