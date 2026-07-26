#include "portfolio_common.h"

// GA – Genetic Algorithm, paper-faithful po Chang et al. (2000): steady-state
// (en otrok na iteracijo zamenja najslabšega, če je boljši), roulette-wheel
// selekcija, uniform crossover na membership vektorju z, Chang-ovo kodiranje
// (Q + s) + water-filling ga_decode.
//   Chang, T.-J., Meade, N., Beasley, J.E. & Sharaiha, Y.M. (2000). "Heuristics
//   for Cardinality Constrained Portfolio Optimisation." Computers & Operations
//   Research 27(13):1271–1302.

struct GAChromosome {
    std::vector<int>    Q;
    std::vector<double> s;
    std::vector<double> w;
    double fitness;
};

static void ga_decode(GAChromosome& chr, int n, double eps_g, double delta_g,
                       const std::vector<double>& eps_v   = {},
                       const std::vector<double>& delta_v = {}) {
    auto ei = [&](int i){ return (!eps_v.empty()   && i<(int)eps_v.size())   ? eps_v[i]   : eps_g; };
    auto di = [&](int i){ return (!delta_v.empty() && i<(int)delta_v.size()) ? delta_v[i] : delta_g; };
    chr.w.assign(n, 0.0);
    int K = (int)chr.Q.size();
    if (K == 0) return;
    // Vodno-polnilna (water-filling) projekcija na {w: Σw=1, εᵢ≤wᵢ≤δᵢ}.
    // Naivni "clamp-na-δ nato normaliziraj" krši zgornjo mejo (deljenje z
    // vsoto lahko vrne wᵢ>δᵢ), zato deljene proporcije s razporejamo z
    // zapiranjem nasičenih (saturated) assetov in prerazporejanjem presežka.
    double sum_eps = 0.0;
    for (int k = 0; k < K; k++) sum_eps += ei(chr.Q[k]);
    double remaining = std::max(0.0, 1.0 - sum_eps);   // razdeliti nad ε
    std::vector<double> extra(K, 0.0), cap(K), sk(K);
    std::vector<char> saturated(K, 0);
    for (int k = 0; k < K; k++) {
        cap[k] = std::max(0.0, di(chr.Q[k]) - ei(chr.Q[k]));  // manevrski prostor
        sk[k]  = std::max(0.0, chr.s[k]);
    }
    for (int iter = 0; iter < K + 1 && remaining > 1e-12; iter++) {
        double sum_active_s = 0.0; int active = 0;
        for (int k = 0; k < K; k++) if (!saturated[k]) { sum_active_s += sk[k]; active++; }
        if (active == 0) break;
        if (sum_active_s < 1e-15) {           // vsi s≈0 → enakomerno
            for (int k = 0; k < K; k++) if (!saturated[k]) sk[k] = 1.0;
            sum_active_s = (double)active;
        }
        double alloc = remaining; remaining = 0.0;
        for (int k = 0; k < K; k++) {
            if (saturated[k]) continue;
            double give = alloc * sk[k] / sum_active_s;
            if (extra[k] + give >= cap[k] - 1e-15) {
                remaining += (extra[k] + give) - cap[k];  // presežek nazaj v bazen
                extra[k] = cap[k]; saturated[k] = 1;
            } else {
                extra[k] += give;
            }
        }
    }
    for (int k = 0; k < K; k++) chr.w[chr.Q[k]] = ei(chr.Q[k]) + extra[k];
}

static double ga_fitness(GAChromosome& chr, const PortfolioData& pd,
                          double lam, double eps, double delta,
                          const std::vector<double>& eps_v   = {},
                          const std::vector<double>& delta_v = {}) {
    g_eval_count++;
    ga_decode(chr, pd.n, eps, delta, eps_v, delta_v);
    // Redko-zasedeno: neničelne uteži so le na chr.Q (K izbranih) → O(K²).
    double var_term = 0.0;
    for (int a : chr.Q) {
        double wa = chr.w[a];
        const std::vector<double>& ca = pd.cov[a];
        for (int b : chr.Q) var_term += wa * chr.w[b] * ca[b];
    }
    double ret_term = 0.0;
    for (int a : chr.Q) ret_term += chr.w[a] * pd.mu[a];
    chr.fitness = lam * var_term - (1.0 - lam) * ret_term;
    return chr.fitness;
}

static void ga_mutate(GAChromosome& chr, int n, int K,
                       double mu_rate, double mu_sigma, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, mu_sigma);
    for (int k = 0; k < K; k++)
        if (uni(rng) < mu_rate)
            chr.s[k] = std::max(0.0, chr.s[k] + gauss(rng));
    if (uni(rng) < mu_rate) {
        std::vector<int> not_in_Q;
        std::vector<bool> in_Q(n, false);
        for (int i : chr.Q) in_Q[i] = true;
        for (int i = 0; i < n; i++) if (!in_Q[i]) not_in_Q.push_back(i);
        if (!not_in_Q.empty()) {
            std::uniform_int_distribution<int> ri_in(0, K - 1);
            std::uniform_int_distribution<int> ri_out(0, (int)not_in_Q.size() - 1);
            int out_pos = ri_in(rng);
            int new_asset = not_in_Q[ri_out(rng)];
            chr.Q[out_pos] = new_asset;
            std::sort(chr.Q.begin(), chr.Q.end());
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            chr.s[out_pos] = uni01(rng);
        }
    }
}

static int ga_roulette(const std::vector<GAChromosome>& pop, std::mt19937& rng) {
    // Minimiziramo fitness → utež = (max_fit - fit) da boljši dobijo več.
    int NP = (int)pop.size();
    double fmax = pop[0].fitness;
    for (auto& c : pop) fmax = std::max(fmax, c.fitness);
    double total = 0.0;
    std::vector<double> wsum(NP);
    for (int i = 0; i < NP; i++) {
        double w = (fmax - pop[i].fitness) + 1e-9;  // boljši (nižji fit) → večja utež
        total += w; wsum[i] = total;
    }
    std::uniform_real_distribution<double> uni(0.0, total);
    double r = uni(rng);
    for (int i = 0; i < NP; i++) if (r <= wsum[i]) return i;
    return NP - 1;
}

std::vector<double> run_ga(const PortfolioData& pd,
                            const Config& cfg,
                            std::mt19937& rng) {
    int n   = pd.n;
    int NP  = cfg.population_size;
    int K   = std::min(cfg.cardinality_K, n);
    double eps   = cfg.w_min;
    double delta = cfg.w_max;
    double lam   = 1.0 - cfg.risk_parameter;
    double mr    = cfg.ga_mutation_rate;
    double ms    = cfg.ga_mutation_sigma;

    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<GAChromosome> pop(NP);
    for (int p = 0; p < NP; p++) {
        GAChromosome& chr = pop[p];
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        chr.Q.assign(perm.begin(), perm.begin() + K);
        std::sort(chr.Q.begin(), chr.Q.end());
        chr.s.resize(K);
        for (int k = 0; k < K; k++) chr.s[k] = uni(rng);
        ga_fitness(chr, pd, lam, eps, delta, cfg.eps, cfg.delta);
    }

    GAChromosome global_best = pop[0];
    for (auto& c : pop) if (c.fitness < global_best.fitness) global_best = c;

    // Steady-state: skupno število ovrednotenj ≈ NP × num_generations,
    // torej naredimo toliko iteracij (vsaka = 1 otrok = 1 ovrednotenje).
    long long iters = (long long)NP * cfg.num_generations;
    for (long long it = 0; it < iters; it++) {
        int p1 = ga_roulette(pop, rng);
        int p2 = ga_roulette(pop, rng);

        // Uniform crossover na membership z (Chang), nato popravilo na K.
        std::vector<int> z(n, 0);
        std::vector<int> z1(n, 0), z2(n, 0);
        for (int i : pop[p1].Q) z1[i] = 1;
        for (int i : pop[p2].Q) z2[i] = 1;
        for (int i = 0; i < n; i++) z[i] = (uni(rng) < 0.5) ? z1[i] : z2[i];

        std::vector<int> ones, zeros;
        for (int i = 0; i < n; i++) (z[i] ? ones : zeros).push_back(i);
        std::shuffle(zeros.begin(), zeros.end(), rng);
        std::shuffle(ones.begin(),  ones.end(),  rng);
        while ((int)ones.size() > K) { z[ones.back()] = 0; ones.pop_back(); }
        while ((int)ones.size() < K && !zeros.empty()) {
            z[zeros.back()] = 1; ones.push_back(zeros.back()); zeros.pop_back();
        }

        GAChromosome child;
        for (int i = 0; i < n; i++) if (z[i]) child.Q.push_back(i);
        // Podeduj s iz starša 1 kjer se asset ujema, sicer naključno.
        for (int idx : child.Q) {
            double sv = uni(rng);
            for (int k = 0; k < (int)pop[p1].Q.size(); k++)
                if (pop[p1].Q[k] == idx) { sv = pop[p1].s[k]; break; }
            child.s.push_back(sv);
        }
        ga_mutate(child, n, K, mr, ms, rng);
        ga_fitness(child, pd, lam, eps, delta, cfg.eps, cfg.delta);

        // Steady-state replace-worst.
        int worst = 0;
        for (int i = 1; i < NP; i++) if (pop[i].fitness > pop[worst].fitness) worst = i;
        if (child.fitness < pop[worst].fitness) pop[worst] = child;
        if (child.fitness < global_best.fitness) global_best = child;
    }

    ga_decode(global_best, n, eps, delta, cfg.eps, cfg.delta);
    return global_best.w;
}
