#ifndef FAISS_INDEX_IVF_FLAT_PANORAMA_PDX_H
#define FAISS_INDEX_IVF_FLAT_PANORAMA_PDX_H

#include "faiss/IndexIVFFlatPanorama.h"
#include "faiss/invlists/PDXInvertedLists.h"

namespace faiss {

struct IndexIVFFlatPanoramaPDX : IndexIVFFlatPanorama {
    explicit IndexIVFFlatPanoramaPDX(
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
};

} // namespace faiss

#endif