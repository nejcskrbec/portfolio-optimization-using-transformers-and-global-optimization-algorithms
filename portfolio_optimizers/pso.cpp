#include "portfolio_common.h"

// RD – rojenje delcev (PSO), prilagojeno KONP po Curi (2009).
//
// Ohranjeni elementi metode:
//   * delec vsebuje zvezne uteži x in binarni izbor z;
//   * hitrosti se posodabljajo glede na globalno in osebno najboljšo rešitev;
//   * koeficienta vpliva se vzorčita iz U[0,2];
//   * binarni položaj se določi s sigmoidno preslikavo in odmikom alpha;
//   * arrange/repair zagotovi natanko K izbranih sredstev in dopustne uteži;
//   * pri popravljanju se uporablja Cura-jeva problemsko odvisna mera c_i.
//
// Računski spremembi, ki ne spreminjata ciljne funkcije:
//   * c_i izračunamo enkrat na optimizacijsko okno;
//   * varianco izračunamo le nad K aktivnimi sredstvi, zato je kvadratni del O(K^2).
//
// Cura, T. (2009). Particle Swarm Optimization Approach to Portfolio
// Optimization. Nonlinear Analysis: Real World Applications 10(4):2396–2406.
// Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization.

struct Particle {
    std::vector<double> x;
    std::vector<int>    z;
    std::vector<double> vx;
    std::vector<double> vz;
    std::vector<double> Gx;  // osebno najboljši položaj uteži
    std::vector<int>    Gz;  // osebno najboljši izbor
    double fit;
    double best_fit;
};

static double pso_fitness(const Particle& p, const PortfolioData& pd, double lam) {
    g_eval_count++;

    std::vector<int> sel;
    sel.reserve(pd.n);
    for (int i = 0; i < pd.n; ++i)
        if (p.z[i]) sel.push_back(i);

    double var_term = 0.0;
    for (int a : sel) {
        const double wa = p.x[a];
        const std::vector<double>& ca = pd.cov[a];
        for (int b : sel)
            var_term += wa * p.x[b] * ca[b];
    }

    double ret_term = 0.0;
    for (int a : sel)
        ret_term += p.x[a] * pd.mu[a];

    return lam * var_term - (1.0 - lam) * ret_term;
}

static void pso_arrange(Particle& p, const PortfolioData& pd,
                        int K, double wmin, double wmax,
                        const std::vector<double>& c_score, std::mt19937& rng,
                        const std::vector<double>& eps = {},
                        const std::vector<double>& delta = {}) {
    const int n = pd.n;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<int> Q, notQ;
    Q.reserve(K);
    notQ.reserve(std::max(0, n - K));
    for (int i = 0; i < n; ++i) {
        if (p.z[i]) Q.push_back(i);
        else notQ.push_back(i);
    }

    // Če je izbranih premalo sredstev, dodajamo naključno ali po c_i.
    while ((int)Q.size() < K && !notQ.empty()) {
        int chosen_idx;
        if (uni(rng) < 0.5 || notQ.size() == 1) {
            std::uniform_int_distribution<int> ri(0, (int)notQ.size() - 1);
            chosen_idx = ri(rng);
        } else {
            double best_c = -1e18;
            int best_ci = 0;
            for (int ii = 0; ii < (int)notQ.size(); ++ii) {
                const double cv = c_score[notQ[ii]];
                if (cv > best_c) {
                    best_c = cv;
                    best_ci = ii;
                }
            }
            chosen_idx = best_ci;
        }

        std::normal_distribution<double> nd(0.0, (wmax - wmin) * 0.1);
        const int asset = notQ[chosen_idx];
        p.z[asset] = 1;
        p.x[asset] = std::max(wmin, std::min(wmax, wmin + std::abs(nd(rng))));
        Q.push_back(asset);
        notQ.erase(notQ.begin() + chosen_idx);
    }

    // Če je izbranih preveč, odstranjujemo naključno ali po najmanjšem c_i.
    while ((int)Q.size() > K) {
        int chosen_idx;
        if (uni(rng) < 0.5 || Q.size() == 1) {
            std::uniform_int_distribution<int> ri(0, (int)Q.size() - 1);
            chosen_idx = ri(rng);
        } else {
            double worst_c = 1e18;
            int worst_ci = 0;
            for (int ii = 0; ii < (int)Q.size(); ++ii) {
                const double cv = c_score[Q[ii]];
                if (cv < worst_c) {
                    worst_c = cv;
                    worst_ci = ii;
                }
            }
            chosen_idx = worst_ci;
        }

        const int asset = Q[chosen_idx];
        p.z[asset] = 0;
        p.x[asset] = 0.0;
        Q.erase(Q.begin() + chosen_idx);
    }

    normalize_weights(p.x, Q, wmin, wmax, eps, delta);
}

std::vector<double> run_pso(const PortfolioData& pd,
                            const Config& cfg,
                            std::mt19937& rng) {
    const int n = pd.n;
    const int P = cfg.population_size;
    const int K = std::min(cfg.cardinality_K, n);
    const double wmin = cfg.w_min;
    const double wmax = cfg.w_max;
    const double lam = 1.0 - cfg.risk_parameter;
    const double alpha = cfg.pso_alpha;

    // Cura-jeva izbirna mera je pri fiksnih mu, Sigma in lambda nespremenljiva,
    // zato jo predizračunamo enkrat na klic optimizatorja.
    std::vector<double> c_score(n);
    {
        std::vector<double> theta(n), rho(n);
        double Om = 0.0, Ps = 0.0;
        for (int j = 0; j < n; ++j) {
            theta[j] = 1.0 + (1.0 - lam) * pd.mu[j];
            rho[j] = 1.0 + lam *
                std::accumulate(pd.cov[j].begin(), pd.cov[j].end(), 0.0) / n;
            Om = std::min(Om, theta[j]);
            Ps = std::min(Ps, rho[j]);
        }
        Om = -Om;
        Ps = -Ps;
        for (int i = 0; i < n; ++i)
            c_score[i] = (theta[i] + Om) / (rho[i] + Ps + 1e-15);
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> uni2(0.0, 2.0);

    std::vector<Particle> swarm(P);
    for (int p = 0; p < P; ++p) {
        Particle& pt = swarm[p];
        pt.x.assign(n, 0.0);
        pt.z.assign(n, 0);
        pt.vx.assign(n, 0.0);
        pt.vz.assign(n, 0.0);

        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < K; ++k) {
            pt.z[perm[k]] = 1;
            pt.x[perm[k]] = wmin + uni(rng) * (wmax - wmin);
        }

        pso_arrange(pt, pd, K, wmin, wmax, c_score, rng, cfg.eps, cfg.delta);
        pt.fit = pso_fitness(pt, pd, lam);
        pt.best_fit = pt.fit;
        pt.Gx = pt.x;
        pt.Gz = pt.z;
    }

    // Globalni vodja je delec z najboljšo DOSLEJ obiskano osebno rešitvijo.
    int gb_idx = 0;
    for (int p = 1; p < P; ++p)
        if (swarm[p].best_fit < swarm[gb_idx].best_fit)
            gb_idx = p;

    // Ohranjamo obstoječo semantiko konfiguracije: skupni proračun vrednotenj
    // je population_size * num_generations.
    const int max_iter = (int)std::floor(
        (double)(cfg.num_generations * cfg.population_size) / P
    );

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int p = 0; p < P; ++p) {
            Particle& pt = swarm[p];

            // Pomembno: za socialno komponento uporabimo globalno najboljšo
            // DOSLEJ obiskano rešitev Gx/Gz trenutnega vodje.
            const std::vector<double> global_x = swarm[gb_idx].Gx;
            const std::vector<int> global_z = swarm[gb_idx].Gz;

            for (int i = 0; i < n; ++i) {
                const double w1 = uni2(rng);
                const double w2 = uni2(rng);

                pt.vz[i] = pt.vz[i]
                         + w1 * (global_z[i] - pt.z[i])
                         + w2 * (pt.Gz[i] - pt.z[i]);

                const double zeta = pt.z[i] + pt.vz[i];
                const double sigmoid = 1.0 / (1.0 + std::exp(-zeta));
                pt.z[i] = (int)std::round(sigmoid - alpha);
                pt.z[i] = std::max(0, std::min(1, pt.z[i]));

                if (pt.z[i] == 1) {
                    pt.vx[i] = pt.vx[i]
                             + w1 * (global_x[i] - pt.x[i])
                             + w2 * (pt.Gx[i] - pt.x[i]);
                    pt.x[i] += pt.vx[i];
                } else {
                    pt.x[i] = 0.0;
                }
            }

            pso_arrange(pt, pd, K, wmin, wmax, c_score, rng, cfg.eps, cfg.delta);
            pt.fit = pso_fitness(pt, pd, lam);

            if (pt.fit < pt.best_fit) {
                pt.best_fit = pt.fit;
                pt.Gx = pt.x;
                pt.Gz = pt.z;
            }

            if (pt.best_fit < swarm[gb_idx].best_fit)
                gb_idx = p;
        }
    }

    return swarm[gb_idx].Gx;
}
