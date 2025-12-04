#include <faiss/invlists/PDXInvertedLists.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace faiss {

PDXInvertedLists::PDXInvertedLists(
        size_t nlist, 
        size_t code_size, 
        size_t n_levels) 
    : ArrayInvertedLists(nlist, code_size), d(code_size / sizeof(float)), n_levels(n_levels) {
    
    FAISS_THROW_IF_NOT(code_size % sizeof(float) == 0);
    
    // PDX block size: Header (n_levels x 64) + Body (d x 64) floats per batch
    // We store this in code_size for proper memory allocation
    this->code_size = (d + n_levels) * kBatchSize * sizeof(float);
    
    // Initialize actual sizes to 0
    actual_list_sizes.resize(nlist, 0);
}

size_t PDXInvertedLists::list_size(size_t list_no) const {
    assert(list_no < nlist);
    return actual_list_sizes[list_no];
}

// Helper function to transpose and embed metadata into a PDX block
static void compute_pdx_block(
        size_t d, 
        size_t n_levels, 
        size_t level_width, 
        size_t curr_batch_size, 
        const float* input_vectors, 
        uint8_t* pdx_block_out) {

    float* block_floats = reinterpret_cast<float*>(pdx_block_out);
    
    // Body starts after the Header (n_levels * kBatchSize floats)
    float* body_base = block_floats + (n_levels * PDXInvertedLists::kBatchSize);
    
    for (size_t i = 0; i < curr_batch_size; i++) {
        const float* vec = input_vectors + i * d;
        
        // Compute Suffix Sums for Tail Energy calculation
        std::vector<float> suffix_sums(d + 1, 0.0f);
        for (int j = (int)d - 1; j >= 0; j--) {
            suffix_sums[j] = suffix_sums[j + 1] + (vec[j] * vec[j]);
        }

        // Write data (Body) and metadata (Header)
        for (size_t k = 0; k < d; k++) {
            // Write to Body (vector data): [Dim k, V0..V63]
            size_t dim_base_offset = k * PDXInvertedLists::kBatchSize;
            body_base[dim_base_offset + i] = vec[k];
        }

        for (size_t l = 0; l < n_levels; l++) {
            size_t start_dim = l * level_width;
            float x_tail_energy = (start_dim < d) ? sqrt(suffix_sums[start_dim]) : 0.0f;

            // Write to Header (metadata): [Level l, V0..V63]
            size_t level_base_offset = l * PDXInvertedLists::kBatchSize;
            block_floats[level_base_offset + i] = x_tail_energy;
        }
    }
}


size_t PDXInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* entry_ids,
        const uint8_t* code) {
    
    size_t old_size = list_size(list_no);
    
    // Calculate level width
    size_t level_width = (d + n_levels - 1) / n_levels;

    const float* input_vectors = reinterpret_cast<const float*>(code);
    
    // 1. Resize list to accommodate the new entries (needs to be block aligned)
    size_t new_size = old_size + n_entry;
    resize(list_no, new_size);
    
    // 2. Get the pointer to the codes list (now resized)
    uint8_t* list_codes = const_cast<uint8_t*>(this->get_codes(list_no));
    
    // 3. Add vectors one by one to correct positions in PDX blocks
    // This is simpler and correct - can optimize later if needed
    size_t block_size_bytes = (d + n_levels) * kBatchSize * sizeof(float);
    
    for (size_t i = 0; i < n_entry; i++) {
        size_t global_pos = old_size + i;  // Position in the list
        size_t batch_no = global_pos / kBatchSize;
        size_t slot_in_batch = global_pos % kBatchSize;
        
        float* block_floats = reinterpret_cast<float*>(list_codes + batch_no * block_size_bytes);
        float* body_base = block_floats + (n_levels * kBatchSize);
        
        const float* vec = input_vectors + i * d;
        
        // Compute suffix sums for tail energy
        std::vector<float> suffix_sums(d + 1, 0.0f);
        for (int j = (int)d - 1; j >= 0; j--) {
            suffix_sums[j] = suffix_sums[j + 1] + (vec[j] * vec[j]);
        }
        
        // Write vector data to Body (vertical layout)
        for (size_t k = 0; k < d; k++) {
            body_base[k * kBatchSize + slot_in_batch] = vec[k];
        }
        
        // Write tail energies to Header
        for (size_t l = 0; l < n_levels; l++) {
            size_t start_dim = l * level_width;
            float x_tail_energy = (start_dim < d) ? sqrt(suffix_sums[start_dim]) : 0.0f;
            block_floats[l * kBatchSize + slot_in_batch] = x_tail_energy;
        }
    }
    
    // 4. Update IDs
    if (entry_ids) {
        faiss::MaybeOwnedVector<idx_t>& list_ids = this->ids[list_no];
        std::memcpy(list_ids.data() + old_size, entry_ids, n_entry * sizeof(idx_t));
    }
    
    // 5. Update actual size
    actual_list_sizes[list_no] = new_size;

    return n_entry;
}

// ------------------- Accessors and Utilities ----------------------

const uint8_t* PDXInvertedLists::get_single_code(
        size_t list_no, 
        size_t offset) const {
    
    size_t block_no = offset / kBatchSize;
    size_t offset_in_block = offset % kBatchSize;
    
    // block size in floats
    size_t block_size_floats = (d + n_levels) * kBatchSize;
    
    const uint8_t* list_base = get_codes(list_no);
    const uint8_t* block_base = list_base + block_no * block_size_floats * sizeof(float);
    
    // Allocate a temporary buffer for the reconstructed horizontal vector
    float* recons = new float[d]; 

    // Body starts after the Header (n_levels * kBatchSize floats)
    const float* block_floats = reinterpret_cast<const float*>(block_base);
    const float* body_base = block_floats + (n_levels * kBatchSize);

    // Gather the vector's components from the vertical layout (Body)
    for (size_t k = 0; k < d; k++) {
        const float* dim_base = body_base + k * kBatchSize; 
        recons[k] = dim_base[offset_in_block]; 
    }

    return reinterpret_cast<const uint8_t*>(recons);
}

void PDXInvertedLists::release_codes(size_t list_no, const uint8_t* codes_ptr) const {
    // If the codes pointer is the temporary buffer from get_single_code
    if (codes_ptr != get_codes(list_no)) {
        delete[] reinterpret_cast<const float*>(codes_ptr);
    }
    // Otherwise, no-op (base ArrayInvertedLists doesn't allocate on get_codes)
}

void PDXInvertedLists::resize(size_t list_no, size_t new_size) {
    // Round up to next batch boundary
    size_t n_batches = (new_size + kBatchSize - 1) / kBatchSize;
    size_t rounded_size = n_batches * kBatchSize;
    
    // Calculate required bytes: n_batches * block_size_bytes
    size_t block_size_bytes = (d + n_levels) * kBatchSize * sizeof(float);
    size_t required_bytes = n_batches * block_size_bytes;
    
    // Resize the codes vector directly
    codes[list_no].resize(required_bytes);
    ids[list_no].resize(rounded_size);
    
    // Update actual size
    actual_list_sizes[list_no] = new_size;
}

InvertedListsIterator* PDXInvertedLists::get_iterator(
        size_t,
        void*) const {
    FAISS_THROW_MSG("PDXInvertedLists does not support standard iterators");
    return nullptr;
}

} // namespace faiss