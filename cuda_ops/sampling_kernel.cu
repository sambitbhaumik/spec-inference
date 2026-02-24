#include <torch/extension.h>
#include <ATen/ATen.h>

#include <limits>

torch::Tensor kv_update(torch::Tensor cache, torch::Tensor update, int64_t start_pos);
torch::Tensor kv_trim(torch::Tensor cache, int64_t seq_len);

torch::Tensor temperature_scale(torch::Tensor logits, double temperature) {
    return logits / temperature;
}

torch::Tensor topk_filter(torch::Tensor logits, int64_t k) {
    if (k <= 0) {
        return logits;
    }

    k = std::min<int64_t>(k, logits.size(-1));
    auto topk = std::get<0>(logits.topk(k, -1));
    auto min_values = topk.select(-1, k - 1).unsqueeze(-1);
    auto mask = logits < min_values;
    auto neg_inf = -std::numeric_limits<float>::infinity();
    return logits.masked_fill(mask, neg_inf);
}

torch::Tensor topp_filter(torch::Tensor logits, double p) {
    if (p >= 1.0 || logits.dim() != 2) {
        return logits;
    }

    auto sorted = logits.sort(-1, true);
    auto sorted_logits = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted);
    auto probs = at::softmax(sorted_logits, -1);
    auto cumulative = probs.cumsum(-1);
    auto mask = cumulative > p;
    mask.index_put_({at::indexing::Slice(), 0}, false);
    auto mask_scattered = at::zeros_like(mask).scatter(-1, sorted_indices, mask);
    auto neg_inf = -std::numeric_limits<float>::infinity();
    return logits.masked_fill(mask_scattered, neg_inf);
}

torch::Tensor minp_filter(torch::Tensor logits, double p) {
    if (p <= 0.0) {
        return logits;
    }

    auto probs = at::softmax(logits, -1);
    auto max_probs = std::get<0>(probs.max(-1, true));
    auto threshold = p * max_probs;
    auto mask = probs < threshold;
    auto neg_inf = -std::numeric_limits<float>::infinity();
    return logits.masked_fill(mask, neg_inf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("temperature_scale", &temperature_scale, "Temperature scaling");
    m.def("topk_filter", &topk_filter, "Top-k filter");
    m.def("topp_filter", &topp_filter, "Top-p filter (2D only)");
    m.def("minp_filter", &minp_filter, "Min-p filter");
    m.def("kv_update", &kv_update, "KV cache update");
    m.def("kv_trim", &kv_trim, "KV cache trim");
}
