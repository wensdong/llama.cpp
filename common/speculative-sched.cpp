#include "speculative-sched.h"

#include "log.h"

#include <algorithm>
#include <cmath>

common_spec_sched::common_spec_sched(const common_spec_sched_params & params) : params(params) {
    pos.resize(params.n_pos_max);
    rest.resize(params.n_pos_max + 2);
    hist_len.assign(params.n_pos_max + 1, 0);
}

void common_spec_sched::update_accept(const std::vector<float> & p_raw, int32_t n_accepted) {
    // position j (1-based) had a trial iff all previous positions were accepted:
    //   j <= n_accepted           -> success
    //   j == n_accepted + 1 <= l  -> failure
    //   j >  n_accepted + 1       -> never reached, no trial
    const int32_t l       = (int32_t) p_raw.size();
    const int32_t n_trial = std::min(l, n_accepted + 1);

    for (int32_t j = 1; j <= n_trial && j <= params.n_pos_max; ++j) {
        auto & s = pos[j - 1];

        s.n     = s.n     * params.ema_stats + 1.0;
        s.p_sum = s.p_sum * params.ema_stats + p_raw[j - 1];
        s.a_sum = s.a_sum * params.ema_stats + (j <= n_accepted ? 1.0 : 0.0);
    }
}

float common_spec_sched::calibrate(int32_t j, float p) const {
    j = std::clamp(j, 1, params.n_pos_max);

    const auto & s = pos[j - 1];
    if (s.n < params.n_warmup || s.p_sum <= 0.0) {
        return p;
    }

    // multiplicative recalibration towards the empirical conditional acceptance
    // rate at this position; order-preserving in p
    const double scale = s.a_sum / s.p_sum;

    return (float) std::clamp(p * scale, 0.0, 1.0);
}

float common_spec_sched::expected_c(int32_t j) const {
    j = std::clamp(j, 1, params.n_pos_max);

    const auto & s = pos[j - 1];
    if (s.n < params.n_warmup) {
        // no data at this depth yet - extrapolate from the deepest calibrated
        // position, shrunk per missing level; unmetered optimism here makes the
        // scheduler overdraft into depths whose real cost it has never seen
        for (int32_t i = j - 1; i >= 1; --i) {
            const auto & q = pos[i - 1];
            if (q.n >= params.n_warmup) {
                const double base = std::clamp(q.a_sum / q.n, 0.0, 1.0);
                return (float) std::max(0.05, base * std::pow(0.85, j - i));
            }
        }
        return 0.5f;
    }

    return (float) std::clamp(s.a_sum / s.n, 0.0, 1.0);
}

void common_spec_sched::update_t_draft_step(int64_t t_us) {
    if (t_us <= 0) {
        return;
    }
    t_draft_step_us = n_draft_step == 0 ? (double) t_us
                                        : params.ema_time * t_draft_step_us + (1.0 - params.ema_time) * (double) t_us;
    n_draft_step++;
}

void common_spec_sched::update_t_rest(int32_t n_verify, int64_t t_us) {
    if (t_us <= 0 || n_verify < 1 || n_verify >= (int32_t) rest.size()) {
        return;
    }

    auto & r = rest[n_verify];

    // reject outliers (request boundaries, scheduling hiccups) once settled
    if (r.n >= 8 && (double) t_us > 5.0 * r.t_us) {
        return;
    }

    r.t_us = r.n == 0 ? (double) t_us
                      : params.ema_time * r.t_us + (1.0 - params.ema_time) * (double) t_us;
    r.n++;
}

double common_spec_sched::t_draft_step() const {
    return t_draft_step_us;
}

double common_spec_sched::t_rest(int32_t n_verify) const {
    n_verify = std::clamp(n_verify, 1, (int32_t) rest.size() - 1);

    if (rest[n_verify].n > 0) {
        return rest[n_verify].t_us;
    }

    // nearest measured buckets below and above
    int32_t lo = -1;
    int32_t hi = -1;
    for (int32_t i = n_verify - 1; i >= 1; --i) {
        if (rest[i].n > 0) { lo = i; break; }
    }
    for (int32_t i = n_verify + 1; i < (int32_t) rest.size(); ++i) {
        if (rest[i].n > 0) { hi = i; break; }
    }

    if (lo >= 0 && hi >= 0) {
        const double w = (double) (n_verify - lo) / (double) (hi - lo);
        return (1.0 - w) * rest[lo].t_us + w * rest[hi].t_us;
    }
    if (lo >= 0) {
        // no data above: extend the slope of the two highest measured buckets
        // (never negative); a flat assumption here proved too optimistic and
        // made the scheduler dive into depths whose true cost it had not seen
        int32_t lo2 = -1;
        for (int32_t i = lo - 1; i >= 1; --i) {
            if (rest[i].n > 0) { lo2 = i; break; }
        }
        double slope = 0.0;
        if (lo2 >= 0) {
            slope = std::max(0.0, (rest[lo].t_us - rest[lo2].t_us) / (double) (lo - lo2));
        }
        return rest[lo].t_us + slope * (n_verify - lo);
    }
    if (hi >= 0) {
        return rest[hi].t_us;
    }

    return 0.0;
}

double common_spec_sched::theta(int32_t l, double sum_a) const {
    // sampling draft token j requires draft decode j, so a submitted chain of
    // length l cost l+1 decodes when it ended on a dropped probe and l when it
    // stopped by lookahead; charge l+1 (the drop-terminated case dominates
    // near the optimum), which also charges the mandatory initial decode to
    // the l = 0 case
    const double t = (double) (l + 1) * t_draft_step_us + t_rest(1 + l);
    if (t <= 0.0) {
        return 0.0;
    }
    return (1.0 + sum_a) / t;
}

common_spec_sched::decision common_spec_sched::decide(int32_t k, float c_k, double a_prev, double sum_prev, int32_t n_max) const {
    const double a_k = a_prev * c_k;

    // at this point k draft decodes have been spent whether token k is kept or
    // dropped - the drop/keep choice only changes the verify batch size, so
    // compare at equal (sunk) draft cost
    const double t_sunk   = (double) k * t_draft_step_us;
    const double th_drop  = (1.0 + sum_prev)       / (t_sunk + t_rest(k));
    const double th_keep  = (1.0 + sum_prev + a_k) / (t_sunk + t_rest(k + 1));

    if (th_keep <= th_drop) {
        return SPEC_SCHED_DROP_STOP;
    }

    if (k >= n_max || k >= params.n_pos_max) {
        return SPEC_SCHED_KEEP_STOP;
    }

    // continuation lookahead: drafting one more token costs one more draft
    // decode; predict its acceptance from the per-position aggregate since the
    // token itself has not been sampled yet (keeps the decision causal)
    const double a_next  = a_k * expected_c(k + 1);
    const double th_next = (1.0 + sum_prev + a_k + a_next) /
                           (t_sunk + t_draft_step_us + t_rest(k + 2));

    if (th_next <= th_keep) {
        return SPEC_SCHED_KEEP_STOP;
    }

    return SPEC_SCHED_CONTINUE;
}

bool common_spec_sched::ready() const {
    return pos[0].n >= params.n_warmup && n_draft_step >= 8 && rest[1].n + rest[2].n + rest[3].n + rest[4].n >= 8;
}

void common_spec_sched::note_len(int32_t l) {
    l = std::clamp(l, 0, (int32_t) hist_len.size() - 1);
    hist_len[l]++;
}

void common_spec_sched::print_stats() const {
    LOG_INF("%s: t_draft_step = %.0f us (n=%lld), ready = %d\n",
            __func__, t_draft_step_us, (long long) n_draft_step, (int) ready());

    for (int32_t j = 1; j <= params.n_pos_max; ++j) {
        const auto & s = pos[j - 1];
        if (s.n < 1.0) {
            break;
        }
        LOG_INF("%s: pos %2d: n = %7.0f, p_pred = %.3f, p_emp = %.3f\n",
                __func__, j, s.n, s.p_sum / s.n, s.a_sum / s.n);
    }

    for (int32_t n = 1; n < (int32_t) rest.size(); ++n) {
        if (rest[n].n > 0) {
            LOG_INF("%s: t_rest(%2d) = %7.0f us (n=%lld)\n", __func__, n, rest[n].t_us, (long long) rest[n].n);
        }
    }

    for (int32_t l = 0; l < (int32_t) hist_len.size(); ++l) {
        if (hist_len[l] > 0) {
            LOG_INF("%s: len %2d: %lld\n", __func__, l, (long long) hist_len[l]);
        }
    }
}
