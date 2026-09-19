#include "portfolio_common.h"

// SO – simulirano ohlajanje po žarilnem okviru Crame in Schynsa (2003),
// prilagojeno naši uteženi povprečje–varianca formulaciji KONP.
//
// Iz članka ohranimo:
//   * Metropolisovo/Boltzmannovo pravilo sprejema;
//   * začetno temperaturo iz povprečnega pozitivnega poslabšanja in v0;
//   * temperaturno stopnjo dolžine sorazmerno n;
//   * geometrijsko ohlajanje;
//   * dva polmera lokalnega premika in ustavitev po stopnjah brez sprejema.
//
// Soseščina je prilagojena našemu problemu. Crama & Schyns uporabljata
// tridelne premike, ki med drugim ohranjajo tudi vnaprej zahtevani pričakovani
// donos. V našem problemu pričakovani donos ni enačbena omejitev, temveč del
// ciljne funkcije, zato ohranjamo proračun in kardinalnost, ne pa fiksnega donosa.
//
// Crama, Y. & Schyns, M. (2003). Simulated Annealing for Complex Portfolio
// Selection Problems. European Journal of Operational Research 150(3):546–571.
// Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by
// Simulated Annealing. Science 220(4598):671–680.

static double sa_objective(const std::vector<double>& x,
                           const PortfolioData& pd, double lam) {
    return mv_objective(x, pd, 1.0 - lam);
}

// Tridelni premik uteži: spremembe treh aktivnih sredstev imajo vsoto nič,
// zato pred projekcijo ohranimo proračunsko omejitev. Polmer omejuje največjo
// absolutno spremembo ene uteži.
static bool sa_three_asset_weight_move(std::vector<double>& x,
                                       const std::vector<int>& active,
                                       double radius,
                                       double wmin, double wmax,
                                       std::mt19937& rng) {
    if (active.size() < 3 || radius <= 0.0) return false;

    std::uniform_int_distribution<int> ri(0, (int)active.size() - 1);
    int ia = ri(rng), ib, ic;
    do { ib = ri(rng); } while (ib == ia);
    do { ic = ri(rng); } while (ic == ia || ic == ib);

    const int a = active[ia], b = active[ib], c = active[ic];

    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    double d1 = ud(rng);
    double d2 = ud(rng);
    double d3 = -(d1 + d2);
    const double max_abs = std::max({std::abs(d1), std::abs(d2), std::abs(d3)});
    if (max_abs < 1e-15) return false;
    d1 /= max_abs;
    d2 /= max_abs;
    d3 /= max_abs;

    double max_scale = radius;
    const auto restrict_scale = [&](int idx, double d) {
        if (d > 1e-15)
            max_scale = std::min(max_scale, (wmax - x[idx]) / d);
        else if (d < -1e-15)
            max_scale = std::min(max_scale, (x[idx] - wmin) / (-d));
    };
    restrict_scale(a, d1);
    restrict_scale(b, d2);
    restrict_scale(c, d3);
    if (!(max_scale > 1e-15) || !std::isfinite(max_scale)) return false;

    std::uniform_real_distribution<double> us(0.0, max_scale);
    const double scale = us(rng);
    x[a] += scale * d1;
    x[b] += scale * d2;
    x[c] += scale * d3;
    return true;
}

// Rezervni dvodelni premik za K<3 ali numerično neizvedljiv tridelni premik.
static void sa_two_asset_weight_move(std::vector<double>& x,
                                     const std::vector<int>& active,
                                     double radius,
                                     std::mt19937& rng) {
    if (active.size() < 2 || radius <= 0.0) return;
    std::uniform_int_distribution<int> ri(0, (int)active.size() - 1);
    int ii = ri(rng), jj;
    do { jj = ri(rng); } while (jj == ii);
    const int a = active[ii], b = active[jj];
    std::uniform_real_distribution<double> step_dist(-radius, radius);
    const double step = step_dist(rng);
    x[a] += step;
    x[b] -= step;
}

static bool sa_replace_asset(std::vector<double>& x,
                             std::vector<int>& active,
                             int n,
                             std::mt19937& rng) {
    if (active.empty() || (int)active.size() >= n) return false;

    std::vector<bool> is_active(n, false);
    for (int a : active) is_active[a] = true;

    std::vector<int> inactive;
    inactive.reserve(n - active.size());
    for (int i = 0; i < n; ++i)
        if (!is_active[i]) inactive.push_back(i);
    if (inactive.empty()) return false;

    std::uniform_int_distribution<int> ro(0, (int)active.size() - 1);
    std::uniform_int_distribution<int> rin(0, (int)inactive.size() - 1);
    const int out_pos = ro(rng);
    const int old_a = active[out_pos];
    const int new_a = inactive[rin(rng)];

    // Prenos celotne uteži ohrani proračun in število aktivnih pozicij.
    const double old_w = x[old_a];
    x[old_a] = 0.0;
    x[new_a] = old_w;
    active[out_pos] = new_a;
    return true;
}

static void sa_neighbor(std::vector<double>& x,
                        std::vector<int>& active,
                        const PortfolioData& pd,
                        const Config& cfg,
                        double radius,
                        std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Majhen delež strukturnih premikov omogoča prehod med različnimi izbirami
    // K sredstev; večina korakov fino preiskuje uteži znotraj trenutne izbire.
    const double replace_prob = 0.10;
    if (uni(rng) < replace_prob && sa_replace_asset(x, active, pd.n, rng)) {
        normalize_weights(x, active, cfg.w_min, cfg.w_max, cfg.eps, cfg.delta);
        return;
    }

    if (!sa_three_asset_weight_move(
            x, active, radius, cfg.w_min, cfg.w_max, rng)) {
        sa_two_asset_weight_move(x, active, radius, rng);
    }
    normalize_weights(x, active, cfg.w_min, cfg.w_max, cfg.eps, cfg.delta);
}

std::vector<double> run_sa(const PortfolioData& pd,
                           const Config& cfg,
                           std::mt19937& rng) {
    const int n = pd.n;
    const int K = std::min(cfg.cardinality_K, n);
    const double wmin = cfg.w_min;
    const double wmax = cfg.w_max;
    const double lam = 1.0 - cfg.risk_parameter;
    const double alpha_cool = cfg.sa_alpha;
    const int L = std::max(1, cfg.sa_stage_length_factor * n);
    const double v0 = cfg.sa_initial_accept_prob;
    const int S_stop = cfg.sa_stop_stages;
    const double q1 = cfg.sa_q1;
    const double q2 = cfg.sa_q2;
    const int q_switch = cfg.sa_q_switch_threshold;

    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Naključna začetna dopustna izbira K sredstev, brez informacij mu/Sigma.
    std::vector<double> x(n, 0.0);
    std::vector<int> active;
    {
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        active.reserve(K);
        for (int k = 0; k < K; ++k) {
            active.push_back(perm[k]);
            x[perm[k]] = 1.0 / K;
        }
    }
    normalize_weights(x, active, wmin, wmax, cfg.eps, cfg.delta);

    double cur_obj = sa_objective(x, pd, lam);
    std::vector<double> best_x = x;
    double best_obj = cur_obj;

    // Kalibracija T0 po Crama–Schyns: poskusni premiki tvorijo začasno verigo
    // in so med kalibracijo sprejeti brez Metropolisovega filtra. Tako vzorčimo
    // tipične pozitivne spremembe ciljne funkcije v dejanski soseščini.
    std::vector<double> tx = x;
    std::vector<int> ta = active;
    double t_obj = cur_obj;
    double sum_delta = 0.0;
    int n_delta = 0;
    for (int step = 0; step < L; ++step) {
        std::vector<double> candidate = tx;
        std::vector<int> cand_active = ta;
        sa_neighbor(candidate, cand_active, pd, cfg, q1, rng);
        const double cand_obj = sa_objective(candidate, pd, lam);
        const double d = cand_obj - t_obj;
        if (d > 0.0) {
            sum_delta += d;
            ++n_delta;
        }
        tx = std::move(candidate);
        ta = std::move(cand_active);
        t_obj = cand_obj;
    }

    const double avg_delta = (n_delta > 0) ? sum_delta / n_delta : 0.01;
    double T = (v0 > 0.0 && v0 < 1.0 && avg_delta > 0.0)
             ? avg_delta / (-std::log(v0))
             : 1.0;

    int stages_no_accept = 0;
    double radius = q1;
    const int max_stages = cfg.num_generations;

    for (int stage = 0; stage < max_stages; ++stage) {
        int accepted = 0;

        for (int step = 0; step < L; ++step) {
            std::vector<double> xt = x;
            std::vector<int> at = active;
            sa_neighbor(xt, at, pd, cfg, radius, rng);

            const double new_obj = sa_objective(xt, pd, lam);
            const double d = new_obj - cur_obj;
            const bool accept =
                (d <= 0.0) || (uni(rng) < std::exp(-d / (T + 1e-15)));

            if (accept) {
                x = std::move(xt);
                active = std::move(at);
                cur_obj = new_obj;
                ++accepted;

                if (cur_obj < best_obj) {
                    best_obj = cur_obj;
                    best_x = x;
                }
            }
        }

        if (accepted < q_switch)
            radius = q2;

        T *= alpha_cool;
        stages_no_accept = (accepted == 0) ? stages_no_accept + 1 : 0;
        if (stages_no_accept >= S_stop)
            break;
    }

    return best_x;
}
