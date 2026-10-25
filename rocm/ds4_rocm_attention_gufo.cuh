// Gufo-derived DeepSeek V4 attention dataflow challenger.
// Adapted from gufo-org/gufo commit 9da89d64b03c13f76085f6221074e927f9d91472
// (src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_attention.hip.hpp), MIT licensed.
// V4.1 launch eligibility remains in ds4_rocm_attention_launch.cuh.

/* The production indexed path uses top_k == 512. Bound this kernel's row table
 * to that contract instead of reserving the generic 1,024-entry capacity. */
#define DS4_ROCM_GUFO_ATTENTION_WMMA_TOPK_CAP 512u
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
template <bool INDEXED, bool COMP_F16>
__global__ __launch_bounds__(1024, 1) static void attention_mixed_heads32_gufo_wmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        float *score_cache,
        uint32_t score_stride,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t HEADS = 32u;
    constexpr uint32_t ROWS = 16u;
    constexpr uint32_t DIM = 512u;
    constexpr uint32_t LDS_DIM = DIM + 4u;

    const uint32_t t = (uint32_t)blockIdx.x;
    const uint32_t head0 = (uint32_t)blockIdx.y * HEADS;
    if (t >= n_tokens || head_dim != DIM || head0 + HEADS > n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    /* The score cache existed so the second pass could skip recomputing QK.
     * The single-pass form below visits each row block once, so nothing reads
     * it; the parameters stay for ABI compatibility with the launchers. */
    (void)score_cache;
    (void)score_stride;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[
        INDEXED ? DS4_ROCM_GUFO_ATTENTION_WMMA_TOPK_CAP : 1u];
    __shared__ uint32_t comp_count_s;
    /* Q is staged transposed, as [k][head], so the score pass reads both of its
     * operands row-major.
     *
     * The score and value passes issue the same number of matrix ops per row
     * block, yet an ablation harness (tools/bench/dsv4_attn_mixed_bench.hip)
     * priced the score pass at about eight times the value pass. Neither
     * spreading it over more waves nor halving its LDS reads moved it, which
     * left the operand layout: K was a col_major matrix_b, and rocwmma has to
     * gather that. With Q transposed the score product becomes
     * `scoresT = KV . Q^T`, both operands row-major, worth 31% of the kernel on
     * the indexed shape and 21% on the mixed window one.
     *
     * The score tile is then [kv row][head] rather than [head][kv row]; every
     * reader below indexes accordingly. */
    constexpr uint32_t QT_PITCH = HEADS + 2u;
    __shared__ half qt_half[DIM * QT_PITCH];
    __shared__ half kv_half[ROWS * LDS_DIM];
    __shared__ float scores[HEADS * ROWS];
    /* Second half of the K split; group 0 writes straight into `scores`. */
    /* Two groups, not four. Four measured -4.4% on this kernel in isolation but
     * nothing end-to-end (433.1 against 433.2 tok/s), and its extra
     * reassociation of the score sum moved the prefill envelope from rmse 0.41
     * to 0.48 -- no longer clearly better than the 16d5e30 baseline's 0.478146.
     * Not worth the margin. */
    constexpr uint32_t QK_WG = 2u;
    __shared__ float qk_part[2u * (QK_WG - 1u) * BM * ROWS];
    __shared__ half probs[HEADS * ROWS];
    __shared__ float softmax_max[HEADS];
    __shared__ float softmax_den[HEADS];
    __shared__ float softmax_rescale[HEADS];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    /* Row-table setup, off the single-thread path.
     *
     * This block used to run entirely on lane 0: a `top_k` loop of 512 *dependent*
     * global reads of `topk`, plus up to 256 serial `raw_rows` writes, once per
     * block -- and there are `n_tokens * n_head / 32` blocks. The window bounds
     * are pure scalar arithmetic, so every thread can derive them without help;
     * `raw_rows` then fills in parallel; and the top-k reads become one coalesced
     * pass into scratch, leaving lane 0 only an LDS-local compaction with no
     * memory latency in the chain. The compaction order and the filter are
     * unchanged, so the row tables are identical. */
    uint32_t raw_count = 0u;
    uint32_t raw_first_idx = 0u;
    if (n_raw != 0u) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    for (uint32_t r = tid; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if constexpr (INDEXED) {
        /* `scores` is untouched until the first row block, so it doubles as the
         * candidate buffer: HEADS * ROWS is exactly the top-k cap. */
        uint32_t *cand = reinterpret_cast<uint32_t *>(scores);
        const uint32_t nk = top_k < DS4_ROCM_GUFO_ATTENTION_WMMA_TOPK_CAP
                                ? top_k : DS4_ROCM_GUFO_ATTENTION_WMMA_TOPK_CAP;
        for (uint32_t i = tid; i < nk; i += blockDim.x) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            cand[i] = (ci >= 0 && (uint32_t)ci < n_comp &&
                       (uint32_t)ci < visible_comp)
                          ? (uint32_t)ci : UINT32_MAX;
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t comp_count = 0u;
            for (uint32_t i = 0u; i < nk; i++) {
                if (cand[i] != UINT32_MAX) comp_rows[comp_count++] = cand[i];
            }
            comp_count_s = comp_count;
        }
    } else if (tid == 0u) {
        comp_count_s = visible_comp;
    }
    __syncthreads();

    for (uint32_t j = tid; j < HEADS * DIM; j += blockDim.x) {
        const uint32_t h = j / DIM;
        const uint32_t d = j - h * DIM;
        qt_half[d * QT_PITCH + h] = __float2half(
            q[((uint64_t)t * n_head + head0 + h) * DIM + d]);
    }
    if (tid < HEADS) {
        softmax_max[tid] = sinks[head0 + tid];
        softmax_den[tid] = 1.0f;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<
        rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b_row = rocwmma::fragment<
        rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<
        rocwmma::accumulator, BM, BN, BK, float>;

    frag_c out0;
    frag_c out1;
    rocwmma::fill_fragment(out0, 0.0f);
    rocwmma::fill_fragment(out1, 0.0f);
    const uint32_t n_score = raw_count + comp_count_s;
    const float score_scale = rsqrtf((float)DIM);
    const uint32_t lane = tid & 31u;
    /* Accumulator element (row, col) for a wave32 16x16 F32 fragment on
     * gfx1151 is (2 * e + lane / 16, lane % 16), recovered with
     * `rocm/tools/wmma_acc_layout.cpp` rather than assumed. `out0` rows are
     * heads 0..15 of the group and `out1` rows heads 16..31, so a per-head
     * factor reaches the accumulator with register arithmetic alone. */
    const uint32_t acc_row_base = lane >> 4u;

    /* Single traversal of the KV rows.
     *
     * The two-pass form walked every row block twice -- once for the scores and
     * the online softmax statistics, once for the probabilities and PV -- and
     * staged, barriered and LDS-wrote the KV tile on both. The score cache
     * spared the second pass its QK matrix multiply but not the staging. Here
     * the running maximum is folded into the accumulator instead: each block
     * rescales the partial output by `exp(m_old - m_new)` before adding its own
     * contribution, and the denominator divides only at the end. Same algebra,
     * one traversal, and the score cache is no longer needed. */
    for (uint32_t row0 = 0u; row0 < n_score; row0 += ROWS) {
        const uint32_t nr = n_score - row0 < ROWS ? n_score - row0 : ROWS;
        /* Stage four values per lane, not one.
         *
         * One half per lane is a 2-byte LDS store: 32 lanes move 64 of the 128
         * bytes the LDS can retire per cycle, and adjacent lanes share banks. The
         * row pitch is a multiple of four halves and `d` is too, so a uint2 store
         * is always 8-byte aligned, and the F32 source is 16-byte aligned for a
         * float4 read. Same values, same locations, a quarter of the accesses. */
        constexpr uint32_t DIM4 = DIM / 4u;
        static_assert(LDS_DIM % 4u == 0u, "row pitch must allow 8-byte stores");
        for (uint32_t j = tid; j < ROWS * DIM4; j += blockDim.x) {
            const uint32_t rr = j / DIM4;
            const uint32_t d = (j - rr * DIM4) * 4u;
            uint2 packed = make_uint2(0u, 0u);
            if (rr < nr) {
                const uint32_t sr = row0 + rr;
                const half *src_h = nullptr;
                const float *src_f = nullptr;
                if (sr < raw_count) {
                    src_f = raw_kv + (uint64_t)raw_rows[sr] * DIM + d;
                } else {
                    uint32_t comp_row = sr - raw_count;
                    if constexpr (INDEXED) {
                        comp_row = comp_rows[comp_row];
                    }
                    if constexpr (COMP_F16) {
                        src_h = ((const half *)comp_kv) + (uint64_t)comp_row * DIM + d;
                    } else {
                        src_f = ((const float *)comp_kv) + (uint64_t)comp_row * DIM + d;
                    }
                }
                if (src_h != nullptr) {
                    packed = *reinterpret_cast<const uint2 *>(src_h);
                } else {
                    const float4 v4 = *reinterpret_cast<const float4 *>(src_f);
                    const __half2 lo = __floats2half2_rn(v4.x, v4.y);
                    const __half2 hi = __floats2half2_rn(v4.z, v4.w);
                    packed.x = *reinterpret_cast<const uint32_t *>(&lo);
                    packed.y = *reinterpret_cast<const uint32_t *>(&hi);
                }
            }
            *reinterpret_cast<uint2 *>(&kv_half[rr * LDS_DIM + d]) = packed;
        }
        __syncthreads();

        constexpr uint32_t QK_WAVES = 2u * QK_WG;
        constexpr uint32_t K_PER_WG = DIM / QK_WG;
        const uint32_t qk_head_block = wave / QK_WG;
        const uint32_t qk_group = wave % QK_WG;
        frag_c score_acc;
        if (wave < QK_WAVES) {
            rocwmma::fill_fragment(score_acc, 0.0f);
            for (uint32_t k0 = qk_group * K_PER_WG;
                 k0 < (qk_group + 1u) * K_PER_WG; k0 += BK) {
                frag_a ka;
                frag_b_row qb;
                rocwmma::load_matrix_sync(ka, kv_half + k0, LDS_DIM);
                rocwmma::load_matrix_sync(
                    qb, qt_half + k0 * QT_PITCH + qk_head_block * BM, QT_PITCH);
                rocwmma::mma_sync(score_acc, ka, qb, score_acc);
            }
            if (qk_group == 0u) {
                rocwmma::store_matrix_sync(scores + qk_head_block * BM,
                                            score_acc, HEADS,
                                            rocwmma::mem_row_major);
            } else {
                rocwmma::store_matrix_sync(
                    qk_part +
                        (qk_head_block * (QK_WG - 1u) + qk_group - 1u) * BM * ROWS,
                    score_acc, ROWS, rocwmma::mem_row_major);
            }
        }
        __syncthreads();
        for (uint32_t j = tid; j < 2u * BM * ROWS; j += blockDim.x) {
            const uint32_t b = j / (BM * ROWS);
            const uint32_t o = j - b * (BM * ROWS);
            float sum = 0.0f;
#pragma unroll
            for (uint32_t gq = 1u; gq < QK_WG; gq++) {
                sum += qk_part[(b * (QK_WG - 1u) + gq - 1u) * BM * ROWS + o];
            }
            const uint32_t r = o / BM;
            const uint32_t hl = o - r * BM;
            scores[r * HEADS + b * BM + hl] += sum;
        }
        __syncthreads();

        /* Advance the online statistics and publish this block's rescale.
         *
         * One wave per head, one row per lane. The obvious form walks the 16
         * rows on 32 of the block's 1024 threads with a carried dependency on
         * both the maximum and the denominator, so the whole workgroup waits on
         * a 16-deep serial chain of `expf`. Reducing across lanes instead needs
         * eight shuffles and keeps every wave busy. Rows beyond `nr` contribute
         * -inf to the maximum and 0 to the sum, so the tail needs no branch. */
        if (wave < HEADS) {
            const float s = lane < nr
                    ? scores[lane * HEADS + wave] * score_scale
                    : -INFINITY;
            float m_new = s;
            for (int off = 16; off > 0; off >>= 1) {
                m_new = fmaxf(m_new, __shfl_xor(m_new, off, 32));
            }
            const float m_old = softmax_max[wave];
            m_new = fmaxf(m_old, m_new);
            float block_sum = lane < nr ? expf(s - m_new) : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                block_sum += __shfl_xor(block_sum, off, 32);
            }
            if (lane == 0u) {
                const float factor = expf(m_old - m_new);
                softmax_den[wave] = softmax_den[wave] * factor + block_sum;
                softmax_max[wave] = m_new;
                softmax_rescale[wave] = factor;
            }
            /* This wave already holds exp(score - m_new) for its own lane while
             * forming the block sum, which is exactly what the separate
             * probability pass recomputed -- so write it here and drop that pass
             * along with the barrier in front of it. Same value, same rounding. */
            if (lane < ROWS) {
                probs[wave * ROWS + lane] =
                    __float2half(lane < nr ? expf(s - m_new) : 0.0f);
            }
        }
        __syncthreads();

        if (row0 != 0u) {
#pragma unroll
            for (uint32_t e = 0u; e < out0.num_elements; e++) {
                const uint32_t r = 2u * e + acc_row_base;
                out0.x[e] *= softmax_rescale[r];
                out1.x[e] *= softmax_rescale[BM + r];
            }
        }

        if (wave < 32u) {
            frag_a p0;
            frag_a p1;
            frag_b_row vb;
            rocwmma::load_matrix_sync(p0, probs, ROWS);
            rocwmma::load_matrix_sync(p1, probs + BM * ROWS, ROWS);
            rocwmma::load_matrix_sync(vb, kv_half + wave * BN, LDS_DIM);
            rocwmma::mma_sync(out0, p0, vb, out0);
            rocwmma::mma_sync(out1, p1, vb, out1);
        }
        __syncthreads();
    }

    if (wave < 32u) {
        /* The denominator was deferred so that no block had to be revisited. */
#pragma unroll
        for (uint32_t e = 0u; e < out0.num_elements; e++) {
            const uint32_t r = 2u * e + acc_row_base;
            const float d0 = softmax_den[r];
            const float d1 = softmax_den[BM + r];
            if (d0 != 0.0f) out0.x[e] /= d0;
            if (d1 != 0.0f) out1.x[e] /= d1;
        }
        float *out = heads + ((uint64_t)t * n_head + head0) * DIM;
        rocwmma::store_matrix_sync(
            out + wave * BN, out0, DIM, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(
            out + BM * DIM + wave * BN,
            out1,
            DIM,
            rocwmma::mem_row_major);
    }
}
#endif
