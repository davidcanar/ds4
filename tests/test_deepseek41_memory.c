/* Exercise actual ROCm admission with controlled host-memory observations. */
#include "../ds4_linux_memory.h"
static uint64_t available;
static bool test_available(uint64_t *bytes) { *bytes = available; return true; }
#define ds4_linux_nonmovable_memory test_available
#include "../ds4.c"
#include <assert.h>

static uint64_t recommended;
uint64_t ds4_gpu_recommended_working_set_size(void) { return recommended; }
int ds4_gpu_stream_expert_cache_get_memory(ds4_gpu_stream_expert_memory *out) {
    memset(out, 0, sizeof(*out));
    return 1;
}

int main(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_engine e = {0};
    const uint8_t support_map = 0;
    ds4_tensor markov_head = {.type = DS4_TENSOR_Q8_0};
    ds4_dspark_weights draft = {0};
    draft.n_stages = 3;
    draft.block_size = 5;
    draft.target_layer_count = 1;
    draft.n_expert = 128;
    draft.n_expert_used = 3;
    draft.markov_rank = 256;
    draft.stage[2].markov_w2 = &markov_head;
    const uint64_t q8_draft_bytes = ds41_draft_bytes(&draft);
    assert(q8_draft_bytes != 0);
    markov_head.type = DS4_TENSOR_F16;
    const uint64_t f16_draft_bytes = ds41_draft_bytes(&draft);
    assert(f16_draft_bytes - q8_draft_bytes ==
           (uint64_t)DS4_N_VOCAB * draft.markov_rank * sizeof(uint16_t));

    /* TP output heads arrive as one packed half per rank.  Expanding the
     * leader half in place must preserve every row before appending its peer. */
    float rows[3 * 8] = {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23};
    const float peer[3 * 4] = {4, 5, 6, 7, 14, 15, 16, 17, 24, 25, 26, 27};
    const float merged[3 * 8] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27};
    ds41_tp_merge_logits_rows(rows, peer, 3, 4);
    assert(memcmp(rows, merged, sizeof(rows)) == 0);

    e.ds41_host_memory_baseline = 180 * gib;
    e.model.size = 152 * gib;
    e.vision_model.size = gib;
    e.vision_ready = e.vision_map_ready = e.ds41_model_loaded = true;
    e.startup_model_span_bytes = e.model.size;
    recommended = 188 * gib;
    /* The observed 192 GB resident failure: loaded weights, 1.5 GiB future
     * graph and 25.9 GiB available. It must tolerate small host fluctuations. */
    for (unsigned mib = 24 * 1024; mib <= 26 * 1024; mib += 64) {
        available = (uint64_t)mib << 20;
        assert(ds41_memory_admit(&e, gib + gib / 2, false));
    }
    /* V4.1 DSpark charges its resident sidecar and exact per-session draft
     * tensors. Before upload, the sidecar is also an outstanding allocation. */
    e.ds41_dspark = true;
    e.dspark_weights = draft;
    e.mtp_model.map = &support_map;
    e.mtp_model.size = 7 * gib;
    e.support_model_loaded = true;
    assert(ds41_engine_graph_bytes(&e, 1024) ==
           ds41_graph_bytes(1024) + f16_draft_bytes);
    recommended = 160 * gib;
    available = 100 * gib;
    assert(!ds41_memory_admit(&e, gib + gib / 2, false));
    e.ds41_dspark = false;
    assert(ds41_memory_admit(&e, gib + gib / 2, false));
    e.ds41_dspark = true;
    recommended = 188 * gib;
    available = 20 * gib;
    e.support_model_loaded = false;
    assert(!ds41_memory_admit(&e, gib, false));
    e.support_model_loaded = true;
    assert(ds41_memory_admit(&e, gib, false));
    e.ds41_dspark = false;
    e.mtp_model.map = NULL;
    e.mtp_model.size = 0;
    /* Existing graph bytes are charged once on restoration/session growth. */
    e.ds41_session_bytes = 2 * gib;
    available = 14 * gib;
    assert(ds41_memory_admit(&e, 2 * gib, false));
    assert(!ds41_memory_admit(&e, 4 * gib, false));
    e.ds41_session_bytes = 0;
    available = 13 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    /* Sidecar upload is still a real outstanding allocation. */
    available = 15 * gib;
    assert(ds41_memory_admit(&e, gib, false));
    e.vision_map_ready = false;
    assert(!ds41_memory_admit(&e, gib, false));
    e.vision_map_ready = true;
    /* The OS minimum and accelerator cap cannot be bypassed. */
    e.ds41_host_memory_baseline = 8 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    e.ds41_host_memory_baseline = 180 * gib;
    recommended = 150 * gib;
    available = 100 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    recommended = 188 * gib;
    assert(!ds41_memory_admit(&e, UINT64_MAX, false));
    /* A 128 GB rank loads only its owned expert half and replicated dense
     * tensors. Both total admission and remaining-allocation checks must
     * charge that same footprint, while keeping the existing reserves. */
    e.ds41_host_memory_baseline = available = 120 * gib;
    recommended = 124 * gib;
    e.startup_model_span_bytes = 0;
    g_tp_shard_model_bytes = 81 * gib;
    assert(ds41_memory_admit(&e, 3 * gib, false));
    available = 93 * gib;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    available = 120 * gib;
    g_tp_shard_model_bytes = 0;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    g_tp_shard_model_bytes = e.startup_model_span_bytes = 81 * gib;
    available = 15 * gib;
    assert(ds41_memory_admit(&e, 3 * gib, false));
    available = 13 * gib;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    g_tp_shard_model_bytes = 0;
    e.ssd_streaming = true;
    assert(ds41_rocm_host_reserve_bytes(128 * gib) == 8 * gib);
    assert(ds41_rocm_stream_reserve_bytes(128 * gib) == 10 * gib);
    assert(ds41_rocm_stream_reserve_bytes(180 * gib) == 13 * gib + gib / 4);
    assert(ds41_rocm_stream_reserve_bytes(0) == 10 * gib);

    /* The resident policy uses only accepted/drafted counters.  It rejects a
     * clearly poor region after four samples and applies the measured C1
    * break-even threshold after eight, without wall-clock input. */
    ds41_draft policy = {0};
    for (int i = 0; i < 3; i++) ds41_draft_adapt(&policy, true, 5, 0, false);
    assert(policy.skip_left == 0 && policy.policy_cycles == 3);
    ds41_draft_adapt(&policy, true, 5, 0, false);
    assert(policy.skip_left == 64 && policy.policy_cycles == 0 &&
           policy.policy_drafted == 0 && policy.policy_accepted == 0);
    memset(&policy, 0, sizeof(policy));
    for (int i = 0; i < 7; i++) ds41_draft_adapt(&policy, true, 5, 2, false);
    assert(policy.skip_left == 0 && policy.policy_cycles == 7);
    ds41_draft_adapt(&policy, true, 5, 2, false);
    assert(policy.skip_left == 64 && policy.policy_cycles == 0);
    memset(&policy, 0, sizeof(policy));
    for (int i = 0; i < 8; i++) ds41_draft_adapt(&policy, true, 5, 3, false);
    assert(policy.skip_left == 0 && policy.policy_cycles == 0 &&
           policy.policy_drafted == 0 && policy.policy_accepted == 0);
    memset(&policy, 0, sizeof(policy));
    for (int i = 0; i < 8; i++)
        ds41_draft_adapt(&policy, true, 1, i < 5, false);
    assert(policy.skip_left == 64 && policy.policy_cycles == 0);
    memset(&policy, 0, sizeof(policy));
    for (int i = 0; i < 4; i++) ds41_draft_adapt(&policy, true, 5, 2, true);
    assert(policy.skip_left == 64 && policy.policy_cycles == 0);
    memset(&policy, 0, sizeof(policy));
    for (int i = 0; i < 4; i++) ds41_draft_adapt(&policy, true, 5, 3, true);
    assert(policy.skip_left == 0 && policy.policy_cycles == 0);
    assert(!ds41_draft_should_propose(true, true, false, 64, false));
    assert(!ds41_draft_should_propose(true, true, true, 64, false));
    assert(ds41_draft_should_propose(true, true, true, 64, true));
    assert(ds41_draft_should_propose(true, true, true, 0, false));
    assert(!ds41_draft_should_propose(false, true, true, 0, false));

    /* Restoring or rewinding a target frontier must not retain positions or
     * policy observations derived from the previous draft frontier. */
    policy.mh_pos0 = 70;
    policy.mh_rows = 7;
    policy.verify_rows = 5;
    policy.skip_left = 19;
    policy.policy_cycles = 3;
    policy.policy_drafted = 12;
    policy.policy_accepted = 8;
    ds41_draft_reset(&policy);
    assert(policy.mh_pos0 == 0 && policy.mh_rows == 0 &&
           policy.verify_rows == 0 && policy.skip_left == 0 &&
           policy.policy_cycles == 0 && policy.policy_drafted == 0 &&
           policy.policy_accepted == 0);

    /* Hardware-aware selection uses cumulative prefix survival and stops at
     * the first throughput decline.  Later logits therefore cannot affect an
     * earlier exclusion. */
    const float all_high[5] = {4, 4, 4, 4, 4};
    const float all_low[5] = {-4, -4, -4, -4, -4};
    const float early_drop_a[5] = {-0.4054651f, -2.1972246f, 8, 8, 8};
    const float early_drop_b[5] = {-0.4054651f, -2.1972246f, -8, -8, -8};
    const float invalid[5] = {4, NAN, 4, 4, 4};
    assert(ds41_draft_scheduled_prefix_len(all_high, 5, false) == 5);
    assert(ds41_draft_scheduled_prefix_len(all_low, 5, false) == 0);
    assert(ds41_draft_scheduled_prefix_len(early_drop_a, 5, false) == 1);
    assert(ds41_draft_scheduled_prefix_len(early_drop_b, 5, false) == 1);
    assert(ds41_draft_scheduled_prefix_len(invalid, 5, false) == 1);
    assert(ds41_draft_scheduled_prefix_len(all_high, 5, true) == 5);
    assert(ds41_draft_scheduled_prefix_len(all_low, 5, true) == 0);
    assert(ds41_draft_scheduled_prefix_len(early_drop_a, 5, true) == 0);
    assert(ds41_draft_scheduled_prefix_len(early_drop_b, 5, true) == 0);
    assert(ds41_draft_scheduled_prefix_len(invalid, 5, true) == 1);
    puts("V4.1 ROCm resident admission, reuse, pressure, sidecar and cap checks PASS");
    return 0;
}
