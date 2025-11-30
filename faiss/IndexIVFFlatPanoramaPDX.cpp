/*
 * PDX-style variant of IndexIVFFlatPanorama.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanoramaPDX.h>

#include <faiss/IndexIVFFlatPanorama.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {

IndexIVFFlatPanoramaPDX::IndexIVFFlatPanoramaPDX(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int n_levels,
        MetricType metric,
        bool own_invlists)
        : IndexIVFFlatPanorama(quantizer, d, nlist, n_levels, metric, own_invlists) {}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanoramaPDX : InvertedListScanner {
    VectorDistance vd;
    const ArrayInvertedListsPanorama* storage;
    using C = typename VectorDistance::C;

    static constexpr size_t kBlockSize = 64;

    IVFFlatScannerPanoramaPDX(
            const VectorDistance& vd,
            const ArrayInvertedListsPanorama* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd), storage(storage) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        cum_sums.resize(storage->n_levels + 1);
    }

    const float* xi = nullptr;
    std::vector<float> cum_sums;
    float q_norm = 0.0f;

    void set_query(const float* query) override {
        this->xi = query;

        const size_t d = vd.d;
        const size_t level_width_floats = storage->level_width / sizeof(float);

        std::vector<float> suffix_sums(d + 1);
        suffix_sums[d] = 0.0f;

        for (int j = (int)d - 1; j >= 0; j--) {
            float squared_val = query[j] * query[j];
            suffix_sums[j] = suffix_sums[j + 1] + squared_val;
        }

        for (size_t level = 0; level < storage->n_levels; level++) {
            size_t start_idx = level * level_width_floats;
            if (start_idx < d) {
                cum_sums[level] = sqrt(suffix_sums[start_idx]);
            } else {
                cum_sums[level] = 0.0f;
            }
        }

        cum_sums[storage->n_levels] = 0.0f;
        q_norm = suffix_sums[0];
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG(
                "IndexIVFFlatPanoramaPDX does not support distance_to_code");
    }

    size_t progressive_filter_batch(
            size_t batch_no,
            size_t list_size,
            const uint8_t* codes_base,
            const float* cum_sums_data,
            float threshold,
            std::vector<float>& exact_distances,
            std::vector<uint32_t>& active_indices,
            const idx_t* ids,
            PanoramaStats& local_stats) const {
        const size_t d = vd.d;
        const size_t level_width_floats = storage->level_width / sizeof(float);

        size_t batch_start = batch_no * storage->kBatchSize;
        size_t curr_batch_size =
                std::min(list_size - batch_start, storage->kBatchSize);

        size_t cumsum_batch_offset =
                batch_no * storage->kBatchSize * (storage->n_levels + 1);
        const float* batch_cum_sums = cum_sums_data + cumsum_batch_offset;

        size_t batch_offset = batch_no * storage->kBatchSize * code_size;
        const uint8_t* storage_base = codes_base + batch_offset;

        // Initialize active set with ID-filtered vectors.
        size_t num_active = 0;
        for (size_t i = 0; i < curr_batch_size; i++) {
            size_t global_idx = batch_start + i;
            bool include = !use_sel || sel->is_member(ids[global_idx]);

            active_indices[num_active] = (uint32_t)i;
            float cum_sum = batch_cum_sums[i];
            exact_distances[i] = cum_sum * cum_sum + q_norm;

            if (include) {
                num_active++;
            }
        }

        if (num_active == 0) {
            return 0;
        }

        size_t total_active = num_active;

        const float* level_cum_sums = batch_cum_sums + storage->kBatchSize;

        // Progressive filtering through levels with a PDX-style vertical kernel.
        for (size_t level = 0; level < storage->n_levels; level++) {
            local_stats.total_dims_scanned += num_active;
            local_stats.total_dims += total_active;

            float query_cum_norm = cum_sums[level + 1];

            size_t level_offset =
                    level * storage->level_width * storage->kBatchSize;
            const float* level_storage =
                    (const float*)(storage_base + level_offset);

            size_t next_active = 0;

            const float* query_level = xi + level * level_width_floats;
            size_t actual_level_width = std::min(
                    level_width_floats, d - level * level_width_floats);

            // Process survivors in blocks; within each block, compute dot
            // products vertically (dimension-first) across survivors.
            for (size_t block_start = 0; block_start < num_active;
                 block_start += kBlockSize) {
                size_t block_size =
                        std::min(kBlockSize, num_active - block_start);

                float local_dots[kBlockSize] = {0.0f};

                for (size_t dim = 0; dim < actual_level_width; dim++) {
                    float qv = query_level[dim];
                    const float* level_dim_base = level_storage + dim;

                    for (size_t b = 0; b < block_size; b++) {
                        uint32_t idx = active_indices[block_start + b];
                        const float* y_ptr =
                                level_dim_base + idx * level_width_floats;
                        local_dots[b] += qv * (*y_ptr);
                    }
                }

                for (size_t b = 0; b < block_size; b++) {
                    uint32_t idx = active_indices[block_start + b];

                    exact_distances[idx] -= 2.0f * local_dots[b];

                    float cum_sum = level_cum_sums[idx];
                    float cauchy_schwarz_bound =
                            2.0f * cum_sum * query_cum_norm;
                    float lower_bound =
                            exact_distances[idx] - cauchy_schwarz_bound;

                    if (C::cmp(threshold, lower_bound)) {
                        active_indices[next_active++] = idx;
                    }
                }
            }

            num_active = next_active;
            if (num_active == 0) {
                break;
            }

            level_cum_sums += storage->kBatchSize;
        }

        return num_active;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;

        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const uint8_t* codes_base = codes;
        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        PanoramaStats local_stats;
        local_stats.reset();

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = progressive_filter_batch(
                    batch_no,
                    list_size,
                    codes_base,
                    cum_sums_data,
                    simi[0],
                    exact_distances,
                    active_indices,
                    ids,
                    local_stats);

            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(simi[0], dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
                }
            }
        }

        indexPanorama_stats.add(local_stats);
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const uint8_t* codes_base = codes;
        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        PanoramaStats local_stats;
        local_stats.reset();

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = progressive_filter_batch(
                    batch_no,
                    list_size,
                    codes_base,
                    cum_sums_data,
                    radius,
                    exact_distances,
                    active_indices,
                    ids,
                    local_stats);

            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    res.add(dis, id);
                }
            }
        }

        indexPanorama_stats.add(local_stats);
    }
};

struct Run_get_InvertedListScannerPDX {
    using T = InvertedListScanner*;

    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlatPanoramaPDX* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        const ArrayInvertedListsPanorama* storage =
                dynamic_cast<const ArrayInvertedListsPanorama*>(ivf->invlists);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "IndexIVFFlatPanoramaPDX requires ArrayInvertedListsPanorama");

        if (sel) {
            return new IVFFlatScannerPanoramaPDX<VD, true>(
                    vd, storage, store_pairs, sel);
        } else {
            return new IVFFlatScannerPanoramaPDX<VD, false>(
                    vd, storage, store_pairs, sel);
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlatPanoramaPDX::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScannerPDX run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

} // namespace faiss

