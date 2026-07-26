#include "portfolio_common.h"

// PSO – Particle Swarm Optimization, paper-faithful po Cura (2009): arrange/repair
// za K-omejitev, ω₁,ω₂ ~ U[0,2] naključno vzorčena vsako iteracijo.
//   Cura, T. (2009). "Particle Swarm Optimization Approach to Portfolio
//   Optimization." Nonlinear Analysis: Real World Applications 10(4):2396–2406.
//   Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." Proc. IEEE
//   Int. Conf. on Neural Networks (ICNN'95), 1942–1948.

struct Particle {
    std::vector<double> x;
    std::vector<int>    z;
    std::vector<double> vx;
    std::vector<double> vz;
    std::vector<double> Gx;
    std::vector<int>    Gz;
    double fit;
    double best_fit;
};

static double pso_fitness(const Particle& p, const PortfolioData& pd, double lam) {
    g_eval_count++;
    // Redko-zasedeno: prispevajo le izbrani (z[i]=1) → O(K²) namesto O(N²).
    std::vector<int> sel;
    for (int i = 0; i < pd.n; i++) if (p.z[i]) sel.push_back(i);
    double var_term = 0.0;
    for (int a : sel) {
        double wa = p.x[a];
        const std::vector<double>& ca = pd.cov[a];
        for (int b : sel) var_term += wa * p.x[b] * ca[b];
    }
    double ret_term = 0.0;
    for (int a : sel) ret_term += p.x[a] * pd.mu[a];
    return lam * var_term - (1.0 - lam) * ret_term;
}

static void pso_arrange(Particle& p, const PortfolioData& pd,
                         int K, double wmin, double wmax,
                         const std::vector<double>& c_score, std::mt19937& rng,
                         const std::vector<double>& eps = {},
                         const std::vector<double>& delta = {}) {
    int n = pd.n;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    // c_score[i] je Cura-jeva izbirna mera, PREDIZRAČUNANA enkrat na klic optimizatorja
    // (v run_pso), ker je odvisna le od μ,Σ,λ (fiksnih v oknu). Prej se je vsakič sproti
    // računala z O(N²) notranjo zanko → arrange O(K·N²); zdaj O(K·N). Vrednosti identične.

    std::vector<int> Q, notQ;
    for (int i = 0; i < n; i++) {
        if (p.z[i]) Q.push_back(i);
        else notQ.push_back(i);
    }

    while ((int)Q.size() < K && !notQ.empty()) {
        int chosen_idx;
        if (uni(rng) < 0.5 || notQ.size() == 1) {
            std::uniform_int_distribution<int> ri(0, (int)notQ.size()-1);
            chosen_idx = ri(rng);
        } else {
            double best_c = -1e18; int best_ci = 0;
            for (int ii = 0; ii < (int)notQ.size(); ii++) {
                double cv = c_score[notQ[ii]];
                if (cv > best_c) { best_c = cv; best_ci = ii; }
            }
            chosen_idx = best_ci;
        }
        std::normal_distribution<double> nd(0.0, (wmax-wmin)*0.1);
        int asset = notQ[chosen_idx];
        p.z[asset] = 1;
        p.x[asset] = std::max(wmin, std::min(wmax, wmin + std::abs(nd(rng))));
        Q.push_back(asset);
        notQ.erase(notQ.begin() + chosen_idx);
    }
    while ((int)Q.size() > K) {
        int chosen_idx;
        if (uni(rng) < 0.5 || Q.size() == 1) {
            std::uniform_int_distribution<int> ri(0, (int)Q.size()-1);
            chosen_idx = ri(rng);
        } else {
            double worst_c = 1e18; int worst_ci = 0;
            for (int ii = 0; ii < (int)Q.size(); ii++) {
                double cv = c_score[Q[ii]];
                if (cv < worst_c) { worst_c = cv; worst_ci = ii; }
            }
            chosen_idx = worst_ci;
        }
        int asset = Q[chosen_idx];
        p.z[asset] = 0;
        p.x[asset] = 0.0;
        Q.erase(Q.begin() + chosen_idx);
    }
    normalize_weights(p.x, Q, wmin, wmax, eps, delta);
}

std::vector<double> run_pso(const PortfolioData& pd,
                             const Config& cfg,
                             std::mt19937& rng) {
    int n  = pd.n;
    int P  = cfg.population_size;
    int K  = std::min(cfg.cardinality_K, n);
    double wmin = cfg.w_min;
    double wmax = cfg.w_max;
    double lam  = 1.0 - cfg.risk_parameter;
    double alpha = cfg.pso_alpha;

    // Predizračun Cura-jeve izbirne mere c_score[i] (odvisna le od μ,Σ,λ, fiksnih v oknu)
    // — enkrat na klic optimizatorja, ne sproti v pso_arrange. Identično originalu.
    std::vector<double> c_score(n);
    {
        std::vector<double> theta(n), rho(n);
        double Om = 0.0, Ps = 0.0;
        for (int j = 0; j < n; j++) {
            theta[j] = 1.0 + (1.0 - lam) * pd.mu[j];
            rho[j]   = 1.0 + lam * std::accumulate(pd.cov[j].begin(), pd.cov[j].end(), 0.0) / n;
            Om = std::min(Om, theta[j]);
            Ps = std::min(Ps, rho[j]);
        }
        Om = -Om; Ps = -Ps;
        for (int i = 0; i < n; i++) c_score[i] = (theta[i] + Om) / (rho[i] + Ps + 1e-15);
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> uni2(0.0, 2.0);

    std::vector<Particle> swarm(P);
    for (int p = 0; p < P; p++) {
        Particle& pt = swarm[p];
        pt.x.assign(n, 0.0);
        pt.z.assign(n, 0);
        pt.vx.assign(n, 0.0);
        pt.vz.assign(n, 0.0);
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < K; k++) {
            pt.z[perm[k]] = 1;
            pt.x[perm[k]] = wmin + uni(rng) * (wmax - wmin);
        }
        pso_arrange(pt, pd, K, wmin, wmax, c_score, rng, cfg.eps, cfg.delta);
        pt.fit = pso_fitness(pt, pd, lam);
        pt.best_fit = pt.fit;
        pt.Gx = pt.x;
        pt.Gz = pt.z;
    }

    int gb_idx = 0;
    for (int p = 1; p < P; p++)
        if (swarm[p].fit < swarm[gb_idx].fit) gb_idx = p;

    int max_iter = (int)std::floor((double)(cfg.num_generations * cfg.population_size) / P);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int p = 0; p < P; p++) {
            Particle& pt = swarm[p];
            for (int i = 0; i < n; i++) {
                double w1 = uni2(rng), w2 = uni2(rng);
                pt.vz[i] = pt.vz[i]
                          + w1 * (swarm[gb_idx].Gz[i] - pt.z[i])
                          + w2 * (pt.Gz[i] - pt.z[i]);
                double zeta = pt.z[i] + pt.vz[i];
                double sigmoid = 1.0 / (1.0 + std::exp(-zeta));
                pt.z[i] = (int)(std::round(sigmoid - alpha));
                pt.z[i] = std::max(0, std::min(1, pt.z[i]));
                if (pt.z[i] == 1) {
                    pt.vx[i] = pt.vx[i]
                              + w1 * (swarm[gb_idx].Gx[i] - pt.x[i])
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
            if (pt.fit < swarm[gb_idx].fit)
                gb_idx = p;
        }
    }

    return swarm[gb_idx].x;
}