#include <torch/extension.h>

torch::Tensor kv_update(torch::Tensor cache, torch::Tensor update, int64_t start_pos) {
    if (cache.dim() != 4 && cache.dim() != 3) {
        throw std::runtime_error("kv_update expects a 3D or 4D cache tensor.");
    }

    int64_t seq_dim = (cache.dim() == 4) ? 2 : 1;
    auto slice = cache.narrow(seq_dim, start_pos, update.size(seq_dim));
    slice.copy_(update);
    return cache;
}

torch::Tensor kv_trim(torch::Tensor cache, int64_t seq_len) {
    if (cache.dim() != 4 && cache.dim() != 3) {
        throw std::runtime_error("kv_trim expects a 3D or 4D cache tensor.");
    }

    int64_t seq_dim = (cache.dim() == 4) ? 2 : 1;
    return cache.narrow(seq_dim, 0, seq_len);
}
