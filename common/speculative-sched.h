#pragma once

#include <cstdint>
#include <vector>

// DSpark-style confidence-scheduled draft-length selection for chained drafters
// (ref: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
// Generation", DeepSeek-AI 2026, sec. 3.2, Algorithm 1 at R=1).
//
// Instead of stopping the draft chain at a static probability threshold (p_min),
// the scheduler picks the draft length that maximizes the expected number of
// generated tokens per second:
//
//   theta(l) = (1 + sum_{j<=l} a_j) / (l * t_draft_step + t_rest(1 + l))
//
// where a_j is the cumulative survival probability of draft position j (product
// of calibrated per-position acceptance estimates) and the denominator is an
// online-profiled cost model of one full decode cycle with l draft tokens.
//
// Every decision is causal: the choice at draft position k depends only on the
// tokens sampled at positions <= k and on cross-request aggregate statistics.
// The rejection-sampling verification itself is untouched, so the target
// distribution is preserved exactly.

struct common_spec_sched_params {
    int32_t n_pos_max = 32;      // max tracked draft position
    int32_t n_warmup  = 50;      // trials before a position's calibration is trusted
    float   ema_stats = 0.995f;  // decay of the per-position calibration statistics
    float   ema_time  = 0.9f;    // decay of the timing estimates
};

struct common_spec_sched {
    explicit common_spec_sched(const common_spec_sched_params & params = {});

    // -- calibration ------------------------------------------------------
    // record the verification outcome of a submitted draft:
    //   p_raw[j] is the raw top-1 draft probability of draft position j+1,
    //   n_accepted is the number of accepted draft tokens (bonus excluded)
    void update_accept(const std::vector<float> & p_raw, int32_t n_accepted);

    // calibrated acceptance estimate for draft position j (1-based) whose token
    // has raw top-1 probability p; order-preserving in p
    float calibrate(int32_t j, float p) const;

    // expected acceptance at position j before its token is sampled (lookahead)
    float expected_c(int32_t j) const;

    // -- cost model -------------------------------------------------------
    void update_t_draft_step(int64_t t_us);              // one chained draft decode
    void update_t_rest(int32_t n_verify, int64_t t_us);  // cycle time outside draft(), by verify batch size

    double t_draft_step() const;
    double t_rest(int32_t n_verify) const;               // interpolated/extrapolated

    // -- decision ---------------------------------------------------------
    // expected tokens/us when submitting a draft of length l with expected
    // accepted-draft count sum_a
    double theta(int32_t l, double sum_a) const;

    enum decision {
        SPEC_SCHED_DROP_STOP = 0, // drop token k, stop drafting
        SPEC_SCHED_KEEP_STOP = 1, // keep token k, stop drafting
        SPEC_SCHED_CONTINUE  = 2, // keep token k, draft another one
    };

    // decide after sampling draft token k (1-based):
    //   c_k      calibrated acceptance estimate of token k
    //   a_prev   cumulative survival product of positions < k
    //   sum_prev sum of survival products of positions < k
    //   n_max    hard cap on the draft length
    decision decide(int32_t k, float c_k, double a_prev, double sum_prev, int32_t n_max) const;

    // true once calibration and timing have enough samples to schedule;
    // callers should fall back to the static p_min rule until then
    bool ready() const;

    // -- observability ----------------------------------------------------
    void note_len(int32_t l);           // record a chosen draft length
    void print_stats() const;           // log calibration table, cost curve, length histogram

    common_spec_sched_params params;

    // per-position statistics (index 0 = position 1)
    struct pos_stats {
        double n     = 0.0; // effective trial count (EMA-weighted)
        double p_sum = 0.0; // weighted sum of raw predicted probabilities
        double a_sum = 0.0; // weighted sum of empirical accepts (0/1)
    };
    std::vector<pos_stats> pos;

    // timing (microseconds)
    double  t_draft_step_us = 0.0;
    int64_t n_draft_step    = 0;

    struct rest_stats {
        double t_us = 0.0;
        int64_t n   = 0;
    };
    std::vector<rest_stats> rest; // indexed by verify batch size (1 + draft length)

    std::vector<int64_t> hist_len; // chosen-length histogram
};
