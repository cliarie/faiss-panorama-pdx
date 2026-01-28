#ifndef FAISS_PDX_PANORAMA_H
#define FAISS_PDX_PANORAMA_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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
};

} // namespace faiss

#endif