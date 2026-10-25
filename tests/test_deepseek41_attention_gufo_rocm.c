/* Focused DeepSeek V4.1 mixed-attention oracle for the Gufo-derived WMMA
 * dataflow.  It exercises production top-k width, ring wrap, invalid/future
 * indices, H32/H64, boundary row counts, allocation canaries, and sampled
 * independent F16-input/F64-reference outputs. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

enum { GUARD_BYTES = 64 };
typedef struct allocation {
    ds4_gpu_tensor *storage, *view;
    size_t bytes;
    struct allocation *next;
} allocation;
static allocation *allocations;

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    allocation *a = calloc(1, sizeof(*a));
    if (!a || bytes > SIZE_MAX - 2 * GUARD_BYTES) { free(a); return NULL; }
    a->bytes = bytes;
    a->storage = ds4_gpu_tensor_alloc(bytes + 2 * GUARD_BYTES);
    if (!a->storage) { free(a); return NULL; }
    unsigned char guard[GUARD_BYTES];
    memset(guard, 0xa5, sizeof(guard));
    if (!ds4_gpu_tensor_write(a->storage, 0, guard, sizeof(guard)) ||
        !ds4_gpu_tensor_write(a->storage, GUARD_BYTES + bytes, guard, sizeof(guard))) goto fail;
    a->view = ds4_gpu_tensor_view(a->storage, GUARD_BYTES, bytes);
    if (!a->view || (data && !ds4_gpu_tensor_write(a->view, 0, data, bytes))) goto fail;
    a->next = allocations;
    allocations = a;
    return a->view;
fail:
    ds4_gpu_tensor_free(a->view);
    ds4_gpu_tensor_free(a->storage);
    free(a);
    return NULL;
}

static int sync_guards(void) {
    CHECK(ds4_gpu_synchronize());
    for (allocation *a = allocations; a; a = a->next) {
        unsigned char before[GUARD_BYTES], after[GUARD_BYTES];
        CHECK(ds4_gpu_tensor_read(a->storage, 0, before, sizeof(before)));
        CHECK(ds4_gpu_tensor_read(a->storage, GUARD_BYTES + a->bytes, after, sizeof(after)));
        for (unsigned i = 0; i < GUARD_BYTES; i++)
            CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    }
    return 1;
}

static void free_all(void) {
    while (allocations) {
        allocation *a = allocations;
        allocations = a->next;
        ds4_gpu_tensor_free(a->view);
        ds4_gpu_tensor_free(a->storage);
        free(a);
    }
}

static uint32_t rng = 0x9e3779b9u;
static float next_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int32_t)(rng % 257u) - 128) / 192.0f;
}
static float f16(float x) { return (float)(_Float16)x; }

static int launch(int indexed,
                  ds4_gpu_tensor *out, const void *model, uint64_t model_bytes,
                  ds4_gpu_tensor *q, ds4_gpu_tensor *raw, ds4_gpu_tensor *comp,
                  ds4_gpu_tensor *topk, uint32_t n_tokens, uint32_t pos0,
                  uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
                  uint32_t n_comp, uint32_t top_k, uint32_t window,
                  uint32_t ratio, uint32_t n_head, uint32_t dim) {
    if (indexed) {
        return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            out, model, model_bytes, 0, q, raw, comp, 0, topk,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head, dim);
    }
    return ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        out, model, model_bytes, 0, q, raw, comp, 0, NULL, 0,
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
        window, ratio, n_head, dim);
}

static double reference_value(const float *q, const float *raw, const float *comp,
                              const int32_t *topk, const float *sinks,
                              uint32_t indexed, uint32_t t, uint32_t head,
                              uint32_t d_out, uint32_t n_tokens, uint32_t pos0,
                              uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
                              uint32_t n_comp, uint32_t top_k, uint32_t window,
                              uint32_t ratio, uint32_t n_head, uint32_t dim) {
    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    const uint32_t last_raw_pos = first_raw_pos + n_raw - 1u;
    uint32_t lo = first_raw_pos;
    if (window && qpos + 1u > window && qpos + 1u - window > lo)
        lo = qpos + 1u - window;
    const uint32_t hi = qpos < last_raw_pos ? qpos : last_raw_pos;
    uint32_t visible_comp = ratio ? (qpos + 1u) / ratio : 0u;
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((size_t)t * n_head + head) * dim;
    double max_score = sinks[head];
    double denom = 0.0;

    double raw_scores[256];
    uint32_t raw_rows[256], nr = 0;
    if (hi >= lo) {
        for (uint32_t pos = lo; pos <= hi && nr < 256u; pos++) {
            const uint32_t row = (raw_start + pos - first_raw_pos) % raw_cap;
            double dot = 0.0;
            for (uint32_t d = 0; d < dim; d++)
                dot += (double)f16(qh[d]) * f16(raw[(size_t)row * dim + d]);
            raw_rows[nr] = row;
            raw_scores[nr] = dot / sqrt((double)dim);
            if (raw_scores[nr] > max_score) max_score = raw_scores[nr];
            nr++;
        }
    }

    double *comp_scores = calloc(top_k ? top_k : visible_comp, sizeof(double));
    uint32_t *comp_rows = calloc(top_k ? top_k : visible_comp, sizeof(uint32_t));
    uint32_t nc = 0;
    const uint32_t limit = indexed ? top_k : visible_comp;
    for (uint32_t i = 0; i < limit; i++) {
        int32_t candidate = indexed ? topk[(size_t)t * top_k + i] : (int32_t)i;
        if (candidate < 0 || (uint32_t)candidate >= visible_comp) continue;
        const uint32_t row = (uint32_t)candidate;
        double dot = 0.0;
        for (uint32_t d = 0; d < dim; d++)
            dot += (double)f16(qh[d]) * f16(comp[(size_t)row * dim + d]);
        comp_rows[nc] = row;
        comp_scores[nc] = dot / sqrt((double)dim);
        if (comp_scores[nc] > max_score) max_score = comp_scores[nc];
        nc++;
    }

    denom = exp((double)sinks[head] - max_score);
    double num = 0.0;
    for (uint32_t i = 0; i < nr; i++) {
        const double p = exp(raw_scores[i] - max_score);
        denom += p;
        num += p * f16(raw[(size_t)raw_rows[i] * dim + d_out]);
    }
    for (uint32_t i = 0; i < nc; i++) {
        const double p = exp(comp_scores[i] - max_score);
        denom += p;
        num += p * f16(comp[(size_t)comp_rows[i] * dim + d_out]);
    }
    free(comp_rows);
    free(comp_scores);
    return denom == 0.0 ? 0.0 : num / denom;
}

static int run_case(uint32_t n_tokens, uint32_t n_head, int indexed) {
    enum { DIM = 512, TOPK = 512, WINDOW = 128, RATIO = 128 };
    const uint32_t pos0 = 73u;
    const uint32_t n_raw = n_tokens + 71u;
    const uint32_t raw_cap = n_raw + 37u;
    const uint32_t raw_start = 19u;
    const uint32_t n_comp = (pos0 + n_tokens + RATIO - 1u) / RATIO + 3u;
    const size_t q_count = (size_t)n_tokens * n_head * DIM;
    const size_t raw_count = (size_t)raw_cap * DIM;
    const size_t comp_count = (size_t)n_comp * DIM;
    const size_t topk_count = (size_t)n_tokens * TOPK;
    float *q = malloc(q_count * sizeof(*q));
    float *raw = malloc(raw_count * sizeof(*raw));
    float *comp = malloc(comp_count * sizeof(*comp));
    int32_t *topk = malloc(topk_count * sizeof(*topk));
    float *control = malloc(q_count * sizeof(*control));
    float *candidate = malloc(q_count * sizeof(*candidate));
    CHECK(q && raw && comp && topk && control && candidate);
    for (size_t i = 0; i < q_count; i++) q[i] = next_value();
    for (size_t i = 0; i < raw_count; i++) raw[i] = next_value();
    for (size_t i = 0; i < comp_count; i++) comp[i] = next_value();
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < TOPK; i++) {
            int32_t v = (int32_t)((i * 17u + t * 3u) % n_comp);
            if (i % 11u == 0u) v = -1;
            else if (i % 13u == 0u) v = (int32_t)n_comp + 5;
            topk[(size_t)t * TOPK + i] = v;
        }
    }

    const size_t page = (size_t)getpagesize();
    void *model = NULL;
    CHECK(posix_memalign(&model, page, page) == 0);
    memset(model, 0, page);
    float *sinks = model;
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.25f + h / 256.0f;
    CHECK(ds4_gpu_set_model_map(model, page));
    ds4_gpu_set_quality(false);
    ds4_gpu_tensor *qt = upload(q, q_count * sizeof(*q));
    ds4_gpu_tensor *rt = upload(raw, raw_count * sizeof(*raw));
    ds4_gpu_tensor *ct = upload(comp, comp_count * sizeof(*comp));
    ds4_gpu_tensor *tt = upload(topk, topk_count * sizeof(*topk));
    ds4_gpu_tensor *ot0 = upload(NULL, q_count * sizeof(*control));
    ds4_gpu_tensor *ot1 = upload(NULL, q_count * sizeof(*candidate));
    CHECK(qt && rt && ct && tt && ot0 && ot1);

    CHECK(setenv("DS4_ROCM_ATTN_GUFO", "0", 1) == 0);
    CHECK(launch(indexed, ot0, model, page, qt, rt, ct, tt, n_tokens, pos0,
                 n_raw, raw_cap, raw_start, n_comp, TOPK, WINDOW, RATIO,
                 n_head, DIM));
    CHECK(sync_guards());
    CHECK(ds4_gpu_tensor_read(ot0, 0, control, q_count * sizeof(*control)));
    CHECK(setenv("DS4_ROCM_ATTN_GUFO", "1", 1) == 0);
    CHECK(launch(indexed, ot1, model, page, qt, rt, ct, tt, n_tokens, pos0,
                 n_raw, raw_cap, raw_start, n_comp, TOPK, WINDOW, RATIO,
                 n_head, DIM));
    CHECK(sync_guards());
    CHECK(ds4_gpu_tensor_read(ot1, 0, candidate, q_count * sizeof(*candidate)));

    double sum_sq = 0.0, max_delta = 0.0, max_ref_error = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < q_count; i++) {
        if (!isfinite(candidate[i])) nonfinite++;
        const double d = (double)candidate[i] - control[i];
        sum_sq += d * d;
        if (fabs(d) > max_delta) max_delta = fabs(d);
    }
    const uint32_t tokens[] = {0u, 1u, n_tokens / 2u, n_tokens - 2u, n_tokens - 1u};
    const uint32_t heads[] = {0u, 15u, 31u, n_head - 1u};
    const uint32_t dims[] = {0u, 1u, 15u, 16u, 127u, 255u, 511u};
    for (size_t ti = 0; ti < sizeof(tokens) / sizeof(tokens[0]); ti++)
        for (size_t hi = 0; hi < sizeof(heads) / sizeof(heads[0]); hi++)
            for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
                const uint32_t t = tokens[ti], h = heads[hi], d = dims[di];
                const double ref = reference_value(q, raw, comp, topk, sinks,
                    indexed, t, h, d, n_tokens, pos0, n_raw, raw_cap, raw_start,
                    n_comp, TOPK, WINDOW, RATIO, n_head, DIM);
                const double error = fabs((double)candidate[((size_t)t * n_head + h) * DIM + d] - ref);
                if (error > max_ref_error) max_ref_error = error;
            }
    const double rmse = sqrt(sum_sq / q_count);
    fprintf(stderr, "attention case rows=%u heads=%u indexed=%d rmse_vs_control=%.9g max_delta=%.9g max_ref_error=%.9g nonfinite=%zu\n",
            n_tokens, n_head, indexed, rmse, max_delta, max_ref_error, nonfinite);
    CHECK(nonfinite == 0);
    if (n_tokens < 128u)
        CHECK(memcmp(control, candidate, q_count * sizeof(*control)) == 0);
    CHECK(max_ref_error < 0.02);
    CHECK(max_delta < 0.04);

    free_all();
    ds4_gpu_cleanup();
    free(model); free(candidate); free(control); free(topk); free(comp); free(raw); free(q);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const uint32_t rows = (uint32_t)strtoul(argv[1], NULL, 10);
    const uint32_t heads = (uint32_t)strtoul(argv[2], NULL, 10);
    const int indexed = atoi(argv[3]);
    if (rows < 2u || rows > 2048u || (heads != 32u && heads != 64u) ||
        (indexed != 0 && indexed != 1)) return 2;
    if (!ds4_gpu_init() || !run_case(rows, heads, indexed) || allocations) return 1;
    puts("DeepSeek V4.1 Gufo attention oracle and canaries PASS");
    return 0;
}
