#ifndef FAISS_INDEX_IVF_FLAT_PDX_H
#define FAISS_INDEX_IVF_FLAT_PDX_H

#include <stdint.h>
#include <cstddef>
#include <faiss/IndexIVFFlat.h>

namespace faiss {

struct IndexIVFFlatPDX : IndexIVFFlat {
    size_t n_levels;

    explicit IndexIVFFlatPDX(
        Index* quantizer, 
        size_t d, 
        size_t nlist, 
        int n_levels,
        MetricType metric = METRIC_L2,
        bool own_invlists = true);
        
    InvertedListScanner* get_InvertedListScanner(
        bool store_pairs, 
        const IDSelector* sel, 
        const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    IndexIVFFlatPDX();
};

} // namespace faiss

#endif