#ifndef FAISS_PDX_INVERTED_LISTS_H
#define FAISS_PDX_INVERTED_LISTS_H

#include <faiss/invlists/InvertedLists.h>

namespace faiss {

/**
 * PDX (Partition Dimensions Across) Inverted Lists
 * 
 * Data layout per batch of 64 vectors:
 *   Header: [n_levels x 64] floats - tail energies for Panorama pruning
 *   Body:   [d x 64] floats - vector components in dimension-major order
 * 
 * This layout enables:
 *   1. Perfect SIMD vectorization (64 contiguous floats per dimension)
 *   2. Cache-efficient vertical scanning
 *   3. Progressive pruning using precomputed tail energies
 */
struct PDXInvertedLists : ArrayInvertedLists {
    static constexpr size_t kBatchSize = 64;
    
    size_t d;           ///< Vector dimension
    size_t n_levels;    ///< Number of Panorama pruning levels
    
    /// Actual number of vectors per list (not rounded to batch boundary)
    std::vector<size_t> actual_list_sizes;

    PDXInvertedLists(size_t nlist, size_t code_size, size_t n_levels);

    size_t list_size(size_t list_no) const override;
    
    size_t add_entries(
            size_t list_no, size_t n_entry,
            const idx_t* ids, const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    const uint8_t* get_single_code(size_t list_no, size_t offset) const override;
    void release_codes(size_t list_no, const uint8_t* codes) const override;
    
    InvertedListsIterator* get_iterator(size_t, void*) const override;
};

} // namespace faiss

#endif