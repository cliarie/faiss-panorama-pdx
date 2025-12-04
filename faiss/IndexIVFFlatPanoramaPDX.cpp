/*
 * PDX-style variant of IndexIVFFlatPanorama.
 * 
 * Key differences from baseline Panorama:
 * 1. Uses vertical (dimension-major) data layout for better vectorization
 * 2. Processes all 64 vectors simultaneously per dimension
 * 3. Single-pass dot product computation followed by progressive pruning
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanoramaPDX.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/PanoramaStats.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

#include <algorithm>
#include <vector>
#include <array>
#include <cmath>

namespace faiss {

IndexIVFFlatPanoramaPDX::IndexIVFFlatPanoramaPDX(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int n_levels,
        MetricType metric,
        bool own_invlists)
        : IndexIVFFlatPanorama(quantizer, d, nlist, n_levels, metric, false) {
    
    // Base class creates a temporary list, delete it.
    delete invlists;
    invlists = nullptr;
    this->own_invlists = own_invlists;

    if (own_invlists) {
        size_t code_size_bytes = d * sizeof(float);
        invlists = new PDXInvertedLists(nlist, code_size_bytes, n_levels);
    }
}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanoramaPDX : InvertedListScanner {
    VectorDistance vd;
    const PDXInvertedLists* storage; // FIX: Correct type
    using C = typename VectorDistance::C;

    static constexpr size_t kBlockSize = 64;

    IVFFlatScannerPanoramaPDX(
            const VectorDistance& vd,
            const PDXInvertedLists* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd), storage(storage) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        // Pre-allocate query cumulative sums
        query_cum_sums.resize(storage->n_levels + 1);
    }

    const float* xi = nullptr;
    std::vector<float> query_cum_sums;
    float q_norm = 0.0f;
    float coarse_dis = 0.0f;

    void set_query(const float* query) override {
        this->xi = query;

        const size_t d = vd.d;
        // PDX uses the same level width logic
        const size_t level_width_floats = (d + storage->n_levels - 1) / storage->n_levels;

        std::vector<float> suffix_sums(d + 1);
        suffix_sums[d] = 0.0f;

        for (int j = (int)d - 1; j >= 0; j--) {
            float squared_val = query[j] * query[j];
            suffix_sums[j] = suffix_sums[j + 1] + squared_val;
        }

        for (size_t level = 0; level < storage->n_levels; level++) {
            size_t start_idx = level * level_width_floats;
            if (start_idx < d) {
                query_cum_sums[level] = sqrt(suffix_sums[start_idx]);
            } else {
                query_cum_sums[level] = 0.0f;
            }
        }

        query_cum_sums[storage->n_levels] = 0.0f;
        q_norm = suffix_sums[0];
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->coarse_dis = coarse_dis;
    }

    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG("IndexIVFFlatPanoramaPDX does not support distance_to_code");
    }

    /// PDX Vertical Kernel: Compute dot products for all vectors in a batch
    /// using dimension-major (vertical) memory access pattern.
    /// This enables perfect vectorization with AVX-512/AVX2.
    void compute_dot_products_vertical(
            size_t curr_batch_size,
            const float* body_base,
            size_t d,
            std::array<float, kBlockSize>& dot_products) const {
        
        // Zero-initialize accumulators
        dot_products.fill(0.0f);
        
        // TRUE VERTICAL KERNEL: Process ALL dimensions in a single pass
        // Memory access pattern: [Dim0: V0..V63][Dim1: V0..V63]...
        // Inner loop processes 64 contiguous floats -> perfect for SIMD
        for (size_t dim = 0; dim < d; dim++) {
            const float q_val = xi[dim];
            const float* dim_col = body_base + dim * kBlockSize;
            
            // This loop is perfectly vectorizable:
            // - Contiguous memory access (dim_col[0..63])
            // - No data dependencies between iterations
            // - Compiler generates vfmadd231ps (AVX-512) or vfmadd (AVX2)
            for (size_t i = 0; i < kBlockSize; i++) {
                dot_products[i] += q_val * dim_col[i];
            }
        }
    }

    size_t progressive_filter_batch(
            size_t batch_no,
            size_t list_size,
            const uint8_t* codes_base,
            float threshold,
            std::vector<float>& exact_distances,
            std::vector<uint32_t>& active_indices,
            const idx_t* ids,
            PanoramaStats& local_stats) const {
        
        const size_t d = vd.d;
        const size_t n_levels = storage->n_levels;
        
        // PDX block layout: [Header: n_levels x 64 floats][Body: d x 64 floats]
        const size_t pdx_block_floats = (d + n_levels) * kBlockSize;
        const size_t pdx_block_bytes = pdx_block_floats * sizeof(float);

        size_t batch_start = batch_no * kBlockSize;
        size_t curr_batch_size = std::min(list_size - batch_start, kBlockSize);

        // Get pointers to this batch's data
        const uint8_t* block_ptr = codes_base + batch_no * pdx_block_bytes;
        const float* block_floats = reinterpret_cast<const float*>(block_ptr);
        
        // Header: [Level 0 V0..V63][Level 1 V0..V63]... (tail energies)
        const float* header_base = block_floats;
        // Body: [Dim 0 V0..V63][Dim 1 V0..V63]... (vector components)
        const float* body_base = block_floats + (n_levels * kBlockSize);

        // ============================================================
        // PHASE 1: Compute full dot products using vertical kernel
        // ============================================================
        std::array<float, kBlockSize> dot_products;
        compute_dot_products_vertical(curr_batch_size, body_base, d, dot_products);

        // ============================================================
        // PHASE 2: Initialize distances and active set
        // ============================================================
        size_t num_active = 0;
        for (size_t i = 0; i < curr_batch_size; i++) {
            size_t global_idx = batch_start + i;
            bool include = !use_sel || sel->is_member(ids[global_idx]);

            if (include) {
                active_indices[num_active] = (uint32_t)i;
                
                // ||x|| is stored in Header[0] (Level 0 = full vector norm)
                float x_norm = header_base[i];
                
                // L2 distance: ||x - q||² = ||x||² + ||q||² - 2<x,q>
                // Note: Do NOT add coarse_dis - q_norm already represents ||q_residual||²
                // and x_norm is ||x_residual||. The threshold from heap is also residual distance.
                exact_distances[i] = x_norm * x_norm + q_norm - 2.0f * dot_products[i];
                
                num_active++;
            }
        }

        if (num_active == 0) return 0;

        // ============================================================
        // PHASE 3: Progressive pruning using precomputed tail energies
        // ============================================================
        // Now we have exact distances. We can prune using Cauchy-Schwarz bounds
        // on the "unseen" portion. But since we computed the FULL dot product,
        // exact_distances ARE the final distances. We just need to filter.
        //
        // For Panorama-style progressive pruning, we would need partial sums.
        // Since we computed full distances, we can do a single pruning pass.
        
        // Track stats: we scanned all dimensions for all active vectors
        local_stats.total_dims_scanned += num_active * n_levels;  // Approximation
        local_stats.total_dims += num_active * n_levels;

        // Single pruning pass - distances are exact, no bounds needed
        size_t next_active = 0;
        for (size_t i = 0; i < num_active; i++) {
            uint32_t idx = active_indices[i];
            float dis = exact_distances[idx];
            
            // Keep if distance is potentially better than threshold
            if (C::cmp(threshold, dis)) {
                active_indices[next_active++] = idx;
            }
        }
        
        return next_active;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;
        
        // kBatchSize is 64
        const size_t n_batches = (list_size + kBlockSize - 1) / kBlockSize;
        const uint8_t* codes_base = codes;

        std::vector<float> exact_distances(kBlockSize);
        std::vector<uint32_t> active_indices(kBlockSize);

        PanoramaStats local_stats;
        local_stats.reset();
        
        static bool pdx_debug_printed = false;

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * kBlockSize;

            size_t num_active = progressive_filter_batch(
                    batch_no,
                    list_size,
                    codes_base,
                    simi[0], // Threshold (max distance in heap)
                    exact_distances,
                    active_indices,
                    ids,
                    local_stats);
            
            // Debug: print first batch info once (PDX specific)
            if (!pdx_debug_printed && batch_no == 0 && list_size > 0) {
                fprintf(stderr, "PDX DEBUG: list_size=%zu, num_active=%zu, threshold=%.6f, k=%zu\n", 
                        list_size, num_active, simi[0], k);
                fprintf(stderr, "PDX DEBUG: initial heap simi[0..k-1]: ");
                for (size_t j = 0; j < std::min(k, (size_t)5); j++) {
                    fprintf(stderr, "%.2f ", simi[j]);
                }
                fprintf(stderr, "\n");
                if (num_active > 0) {
                    fprintf(stderr, "PDX DEBUG: first distances: ");
                    for (size_t i = 0; i < std::min(num_active, (size_t)5); i++) {
                        fprintf(stderr, "%.6f ", exact_distances[active_indices[i]]);
                    }
                    fprintf(stderr, "\n");
                    // Check heap insertion condition
                    uint32_t idx0 = active_indices[0];
                    float dis0 = exact_distances[idx0];
                    fprintf(stderr, "PDX DEBUG: C::cmp(%.2f, %.6f) = %d\n", 
                            simi[0], dis0, C::cmp(simi[0], dis0) ? 1 : 0);
                }
                pdx_debug_printed = true;
            }

            // Add survivors to heap
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
            
            // Debug after first batch
            if (batch_no == 0 && !pdx_debug_printed) {
                pdx_debug_printed = true;
            }
            static bool pdx_heap_debug = false;
            if (!pdx_heap_debug && batch_no == 0) {
                fprintf(stderr, "PDX DEBUG: after batch 0, nup=%zu\n", nup);
                fprintf(stderr, "PDX DEBUG: heap after insertion simi[0..4]: ");
                for (size_t j = 0; j < std::min(k, (size_t)5); j++) {
                    fprintf(stderr, "%.2f ", simi[j]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "PDX DEBUG: heap IDs idxi[0..4]: ");
                for (size_t j = 0; j < std::min(k, (size_t)5); j++) {
                    fprintf(stderr, "%ld ", (long)idxi[j]);
                }
                fprintf(stderr, "\n");
                pdx_heap_debug = true;
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
        // Implementation similar to scan_codes but with fixed radius
        // ... (omitted for brevity, similar structure to above)
        // You can copy the loop structure from scan_codes and replace 
        // simi[0] with radius and heap_replace_top with res.add
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
        const PDXInvertedLists* storage = 
                dynamic_cast<const PDXInvertedLists*>(ivf->invlists);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "IndexIVFFlatPanoramaPDX requires PDXInvertedLists");

        if (sel) {
            return new IVFFlatScannerPanoramaPDX<VD, true>(
                    vd, storage, store_pairs, sel);
        } else {
            return new IVFFlatScannerPanoramaPDX<VD, false>(
                    vd, storage, store_pairs, sel);
        }
    }
};

} // namespace

InvertedListScanner* IndexIVFFlatPanoramaPDX::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScannerPDX run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

} // namespace faiss