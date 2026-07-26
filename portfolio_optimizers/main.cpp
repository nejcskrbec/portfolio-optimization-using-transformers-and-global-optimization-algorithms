#include "portfolio_common.h"
#include "pso.cpp"
#include "sa.cpp"
#include "ga.cpp"

int main(int argc, char* argv[]) {
    std::string cfg_file = "config.json";
    if (argc > 1) cfg_file = argv[1];

    std::ifstream cfg_ifs(cfg_file);
    if (!cfg_ifs.is_open()) {
        std::cerr << "Napaka: Ne morem odpreti " << cfg_file << "\n";
        return 1;
    }
    std::string cfg_content((std::istreambuf_iterator<char>(cfg_ifs)),
                              std::istreambuf_iterator<char>());
    cfg_ifs.close();

    // ── Preberi data_bridge.json ─────────────────────────────
    // Vsebuje: "tickers", "mu" (transformer napovedi), opcijsko "cov"
    // Cov se sicer izračuna v run_benchmark.py in posreduje sem direktno,
    // zato je "cov" v data_bridge.json opcijsko (za standalone zagon).
    std::string bridge_file = get_json_val(cfg_content, "data_bridge_file");
    if (bridge_file.empty()) bridge_file = "data_bridge.json";

    std::string bridge_content = "";
    {
        std::ifstream bfs(bridge_file);
        if (bfs.is_open())
            bridge_content = std::string((std::istreambuf_iterator<char>(bfs)),
                                         std::istreambuf_iterator<char>());
    }

    // ── Napolni Config ───────────────────────────────────────
    Config cfg;
    cfg.optimizer_type = get_json_val(cfg_content, "optimizer_type");
    if (cfg.optimizer_type.empty()) cfg.optimizer_type = "all";

    std::string s_common = get_section(cfg_content, "common");
    cfg.population_size = get_int(s_common, "population_size", 100);
    cfg.num_generations = get_int(s_common, "num_generations", 1000);
    cfg.risk_parameter  = get_dbl(s_common, "risk_parameter",  0.5);
    cfg.cardinality_K   = get_int(s_common, "cardinality_K",   10);
    cfg.w_min           = get_dbl(s_common, "w_min",           0.01);
    cfg.w_max           = get_dbl(s_common, "w_max",           0.95);
    cfg.seed            = (unsigned int)get_int(s_common, "seed", 42);

    std::string s_pso = get_section(cfg_content, "pso");
    // Samo Cura (2009) α — ω₁,ω₂ ~ U[0,2] se vzorčijo v pso.cpp.
    cfg.pso_alpha = get_dbl(s_pso, "alpha", 0.06);

    std::string s_sa = get_section(cfg_content, "sa");
    cfg.sa_alpha               = get_dbl(s_sa, "alpha",               0.95);
    cfg.sa_stage_length_factor = get_int(s_sa, "stage_length_factor", 2);
    cfg.sa_initial_accept_prob = get_dbl(s_sa, "initial_accept_prob", 0.8);
    cfg.sa_stop_stages         = get_int(s_sa, "stop_stages",         5);
    cfg.sa_q1                  = get_dbl(s_sa, "q1",                  0.005);
    cfg.sa_q2                  = get_dbl(s_sa, "q2",                  0.001);
    cfg.sa_q_switch_threshold  = get_int(s_sa, "q_switch_threshold",  10);

    std::string s_ga = get_section(cfg_content, "ga");
    cfg.ga_mutation_rate   = get_dbl(s_ga, "mutation_rate",   0.01);
    cfg.ga_mutation_sigma  = get_dbl(s_ga, "mutation_sigma",  0.05);

    PortfolioData pd;

    // Tickers se berejo iz bridge.json (avail_tickers iz yfinance — abecedni vrstni red)
    // NE iz config.json kjer je vrstni red poljuben — sicer pride do μ/ticker mismatch!
    std::string tickers_str = get_json_val(bridge_content, "tickers");
    if (tickers_str.empty())
        tickers_str = get_json_val(cfg_content, "tickers");  // fallback za standalone
    pd.tickers = parse_list_str(tickers_str);
    pd.n = (int)pd.tickers.size();
    if (pd.n == 0) {
        std::cerr << "Napaka: Ni tikerjev v bridge.json ali config.json\n";
        return 1;
    }

    // mu: iz data_bridge.json
    std::string mu_str = get_json_val(bridge_content, "mu");
    if (!mu_str.empty())
        pd.mu = parse_list_dbl(mu_str);
    if ((int)pd.mu.size() != pd.n) {
        std::cerr << "Napaka: mu v data_bridge.json ne ustreza stevilu tikerjev ("
                  << pd.mu.size() << " != " << pd.n << ")\n";
        return 1;
    }

    // cov: obvezno iz data_bridge.json — run_benchmark.py ga vedno pošlje
    std::string cov_str = get_json_val(bridge_content, "cov");
    if (cov_str.empty()) {
        std::cerr << "Napaka: Kovariancna matrika ni najdena v " << bridge_file << ".\n";
        return 1;
    }
    std::vector<double> cov_flat = parse_list_dbl(cov_str);
    if ((int)cov_flat.size() != pd.n * pd.n) {
        std::cerr << "Napaka: Pricakujem " << pd.n * pd.n
                  << " elementov v \"cov\", dobil " << cov_flat.size() << ".\n";
        return 1;
    }
    pd.cov.assign(pd.n, std::vector<double>(pd.n));
    for (int i = 0; i < pd.n; i++)
        for (int j = 0; j < pd.n; j++)
            pd.cov[i][j] = cov_flat[i * pd.n + j];

    // ── Per-asset εᵢ in δᵢ (Chang et al. 2000, Eq. 12) ──────────────────
    // Bere iz bridge.json: "eps": [ε₁,...,εN], "delta": [δ₁,...,δN]
    // Fallback na globalen cfg.w_min / cfg.w_max če ni podanih.
    std::string eps_str   = get_json_val(bridge_content, "eps");
    std::string delta_str = get_json_val(bridge_content, "delta");
    if (!eps_str.empty()) {
        cfg.eps = parse_list_dbl(eps_str);
        if ((int)cfg.eps.size() != pd.n) cfg.eps.clear();
    }
    if (!delta_str.empty()) {
        cfg.delta = parse_list_dbl(delta_str);
        if ((int)cfg.delta.size() != pd.n) cfg.delta.clear();
    }

    // cov_selection se ignorira — kardinalnost je integrirana v algoritme

    // ── Izpis ────────────────────────────────────────────────
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout <<   "║        PORTFELJSKA OPTIMIZACIJA – Magistrska naloga  ║\n";
    std::cout <<   "╚══════════════════════════════════════════════════════╝\n";
    std::cout << "Stevilo sredstev: " << pd.n << "\n";
    std::cout << "Algoritem: " << cfg.optimizer_type << "\n";
    std::cout << "Parameter tveganja P: " << cfg.risk_parameter << "\n";
    std::cout << "Kardinalnost K: " << cfg.cardinality_K << "\n";
    std::cout << "Populacija: " << cfg.population_size
              << "  Generacije: " << cfg.num_generations << "\n\n";

    // ── Vsak algoritem sam rešuje kardinalnost K ────────────────
    // Ni zunanje predselekcije — algoritmi prejmejo cel problem (n assetov)
    // in interno izberejo K optimalnih assetov skupaj z utežmi.
    // show() izpiše portfelj + število ovrednotenj ciljne funkcije (budget).
    auto show = [&](const std::string& name, const std::vector<double>& w) {
        double ret  = mv_return(w, pd);
        double risk = mv_risk(w, pd);
        print_result(name, pd.tickers, w, ret, risk, cfg.risk_parameter);
        std::cout << "  Ovrednotenj ciljne funkcije: " << g_eval_count << "\n";
    };
    // run() resetira budget-števec pred vsakim algoritmom → poštena primerjava.
    auto run_algo = [&](const std::string& label, unsigned int seed_off,
                        std::vector<double>(*fn)(const PortfolioData&,
                                                 const Config&, std::mt19937&)) {
        g_eval_count = 0;
        std::mt19937 rng(cfg.seed + seed_off);
        auto w = fn(pd, cfg, rng);
        show(label, w);
    };

    std::string algo = cfg.optimizer_type;
    std::transform(algo.begin(), algo.end(), algo.begin(), ::tolower);

    if (algo == "pso" || algo == "all") {
        std::cout << "\n[PSO] Zaganjam (Cura 2009)...\n";
        run_algo("PSO (Cura 2009)", 2, run_pso);
    }
    if (algo == "sa" || algo == "all") {
        std::cout << "\n[SA] Zaganjam paper-faithful SA (Crama & Schyns 2003)...\n";
        run_algo("SA – Simulated Annealing (Crama & Schyns 2003)", 3, run_sa);
    }
    if (algo == "ga" || algo == "all") {
        std::cout << "\n[GA] Zaganjam paper-faithful steady-state GA (Chang et al. 2000)...\n";
        run_algo("GA – Genetic Algorithm (Chang et al. 2000)", 4, run_ga);
    }

    std::cout << "\nOptimizacija zakljucena.\n";
    return 0;
}