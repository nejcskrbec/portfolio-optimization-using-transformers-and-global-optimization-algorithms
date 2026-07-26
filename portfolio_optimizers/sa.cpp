#include "portfolio_common.h"

// SA – Simulated Annealing, paper-faithful po Crama & Schyns (2003). Poteza je
// čisto naključna: (a) prerazporeditev uteži med dvema aktivnima assetoma ali
// (b) naključna zamenjava aktivnega z neaktivnim — brez μ/σ (Sharpe) informacije
// v iskalni potezi. Žarilni okvir + kalibracija T iz povprečne pozitivne Δ.
//   Crama, Y. & Schyns, M. (2003). "Simulated Annealing for Complex Portfolio
//   Selection Problems." European Journal of Operational Research 150(3):546–571.
//   Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). "Optimization by
//   Simulated Annealing." Science 220(4598):671–680.

static double sa_objective(const std::vector<double>& x,
                            const PortfolioData& pd, double lam) {
    return mv_objective(x, pd, 1.0 - lam);
}

static void sa_move(std::vector<double>& x, const PortfolioData& pd,
                     const std::vector<int>& active, double radius,
                     std::mt19937& rng) {
    if (active.size() < 2) return;
    std::uniform_int_distribution<int> ri(0, (int)active.size()-1);
    int ii = ri(rng), jj;
    do { jj = ri(rng); } while (jj == ii);
    int a1 = active[ii], a2 = active[jj];
    std::uniform_real_distribution<double> sign_d(-1.0, 1.0);
    double step = sign_d(rng) * radius;
    x[a1] += step;
    x[a2] -= step;
}

std::vector<double> run_sa(const PortfolioData& pd,
                            const Config& cfg,
                            std::mt19937& rng) {
    int n  = pd.n;
    int K  = std::min(cfg.cardinality_K, n);
    double wmin = cfg.w_min;
    double wmax = cfg.w_max;
    double lam  = 1.0 - cfg.risk_parameter;
    double alpha_cool = cfg.sa_alpha;
    int    L = cfg.sa_stage_length_factor * n;
    double v0 = cfg.sa_initial_accept_prob;
    int    S_stop = cfg.sa_stop_stages;
    double q1 = cfg.sa_q1, q2 = cfg.sa_q2;
    int    q_switch = cfg.sa_q_switch_threshold;

    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Naključna začetna izbira K assetov (brez μ-sortiranja — faithful).
    std::vector<double> x(n, 0.0);
    std::vector<int>    active;
    {
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < K; k++) { active.push_back(perm[k]); x[perm[k]] = 1.0 / K; }
    }
    normalize_weights(x, active, wmin, wmax, cfg.eps, cfg.delta);

    double cur_obj = sa_objective(x, pd, lam);
    std::vector<double> best_x = x;
    double best_obj = cur_obj;

    // Kalibracija T iz povprečne pozitivne Δ (Crama & Schyns 2003).
    double sum_delta = 0.0; int n_delta = 0;
    for (int step = 0; step < L; step++) {
        std::vector<double> xt = x;
        sa_move(xt, pd, active, q1, rng);
        normalize_weights(xt, active, wmin, wmax, cfg.eps, cfg.delta);
        double d = sa_objective(xt, pd, lam) - cur_obj;
        if (d > 0.0) { sum_delta += d; n_delta++; }
    }
    double avg_delta = (n_delta > 0) ? sum_delta / n_delta : 0.01;
    double T = (v0 > 0.0 && v0 < 1.0 && avg_delta > 0.0)
               ? avg_delta / (-std::log(v0)) : 1.0;

    int stages_no_accept = 0;
    double radius = q1;
    int max_stages = cfg.num_generations;

    for (int stage = 0; stage < max_stages; stage++) {
        int accepted = 0;
        for (int step = 0; step < L; step++) {
            std::vector<double> xt = x;
            std::vector<int>    at = active;

            if (uni(rng) < 0.10) {
                // Čisto naključna zamenjava aktiven ↔ neaktiven asset.
                std::vector<int> inactive;
                std::vector<bool> is_act(n, false);
                for (int a : at) is_act[a] = true;
                for (int i = 0; i < n; i++) if (!is_act[i]) inactive.push_back(i);
                if (!inactive.empty()) {
                    std::uniform_int_distribution<int> ro(0, (int)at.size()-1);
                    std::uniform_int_distribution<int> rin(0, (int)inactive.size()-1);
                    int out_pos = ro(rng);
                    int new_a   = inactive[rin(rng)];
                    xt[at[out_pos]] = 0.0;
                    xt[new_a] = wmin;
                    at[out_pos] = new_a;
                    normalize_weights(xt, at, wmin, wmax, cfg.eps, cfg.delta);
                }
            } else {
                sa_move(xt, pd, at, radius, rng);
                normalize_weights(xt, at, wmin, wmax, cfg.eps, cfg.delta);
            }

            double d = sa_objective(xt, pd, lam) - cur_obj;
            bool accept = (d < 0.0) || (uni(rng) < std::exp(-d / (T + 1e-15)));
            if (accept) {
                x = xt; active = at; cur_obj += d; accepted++;
                if (cur_obj < best_obj) { best_obj = cur_obj; best_x = x; }
            }
        }
        if (accepted < q_switch) radius = q2;
        T *= alpha_cool;
        stages_no_accept = (accepted == 0) ? stages_no_accept + 1 : 0;
        if (stages_no_accept >= S_stop) break;
    }

    return best_x;
}
