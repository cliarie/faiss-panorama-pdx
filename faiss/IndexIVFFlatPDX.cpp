#include <faiss/IndexIVFFlatPDX.h>

#include <cstring>
#include <cstdio>

#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/invlists/InvertedLists.h>

#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexIVFFlatPDX::IndexIVFFlatPDX(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int n_levels,
        MetricType metric,
        bool own_invlists)
        : IndexIVFFlat(quantizer, d, nlist, metric, false), n_levels(n_levels) {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);

    this->invlists = new ArrayInvertedListsPDX(nlist, code_size, n_levels);
    this->own_invlists = own_invlists;
}

IndexIVFFlatPDX::IndexIVFFlatPDX() : n_levels(0) {}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPDX : InvertedListScanner {
    VectorDistance vd;
    const ArrayInvertedListsPDX* storage;
    using C = typename VectorDistance::C;
    static constexpr MetricType metric = VectorDistance::metric;

    IVFFlatScannerPDX(
            const VectorDistance& vd,
            const ArrayInvertedListsPDX* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd), storage(storage) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        cum_sums.resize(storage->n_levels + 1);
    }

    const float* xi = nullptr;
    std::vector<float> cum_sums;

    void set_query(const float* query) override {
        this->xi = query;
        this->storage->pano.compute_query_cum_sums(query, cum_sums.data());
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG("IndexIVFFlatPDX does not support distance_to_code");
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

        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = with_metric_type(metric, [&]<MetricType M>() {
                return storage->pdx.progressive_filter_batch<C, M>(
                        codes,
                        cum_sums_data,
                        xi,
                        cum_sums.data(),
                        batch_no,
                        list_size,
                        sel,
                        ids,
                        use_sel,
                        active_indices,
                        exact_distances,
                        simi[0]);
            });

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

        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = with_metric_type(metric, [&]<MetricType M>() {
                return storage->pdx.progressive_filter_batch<C, M>(
                        codes,
                        cum_sums_data,
                        xi,
                        cum_sums.data(),
                        batch_no,
                        list_size,
                        sel,
                        ids,
                        use_sel,
                        active_indices,
                        exact_distances,
                        radius);
            });

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
    }
};

struct Run_get_InvertedListScanner {
    using T = InvertedListScanner*;

    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlatPDX* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        const ArrayInvertedListsPDX* storage =
                dynamic_cast<const ArrayInvertedListsPDX*>(ivf->invlists);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "IndexIVFFlatPDX requires ArrayInvertedListsPDX");

        if (sel) {
            return new IVFFlatScannerPDX<VD, true>(vd, storage, store_pairs, sel);
        } else {
            return new IVFFlatScannerPDX<VD, false>(
                    vd, storage, store_pairs, sel);
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlatPDX::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScanner run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

void IndexIVFFlatPDX::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);
    memcpy(recons, code, code_size);
    invlists->release_codes(list_no, code);
}

} // namespace faiss