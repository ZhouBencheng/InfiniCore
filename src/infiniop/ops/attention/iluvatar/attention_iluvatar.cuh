#ifndef __INFINIOP_ATTENTION_ILUVATAR_CUH__
#define __INFINIOP_ATTENTION_ILUVATAR_CUH__

#include "infinicore.h"

namespace op::attention::iluvatar {

infiniStatus_t fused_attention(
    infiniDtype_t dtype,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    void *k_cache,
    void *v_cache,
    size_t n_head,
    size_t seq_len,
    size_t head_dim,
    size_t total_seq_len,
    size_t cache_len,
    size_t pos,
    void *stream);

} // namespace op::attention::iluvatar

#endif // __INFINIOP_ATTENTION_ILUVATAR_CUH__
