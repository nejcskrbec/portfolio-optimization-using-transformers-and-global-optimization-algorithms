#include "portfolio_common.h"
#include "pso.cpp"
#include "sa.cpp"

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
    // Vsebuje: "tickers", "mu" (transformerske napovedi) in "cov".
    std::string bridge_file = get_json_val(cfg_content, "data_bridge_file");
    if (bridge_file.empty()) bridge_file = "data_bridge.json";

    std::string bridge_content;
    {
        std::ifstream bfs(bridge_file);
        if (bfs.is_open()) {
            bridge_content = std::string(
                (std::istreambuf_iterator<char>(bfs)),
                std::istreambuf_iterator<char>()
            );
        }
    }

    // ── Napolni Config ───────────────────────────────────────
    Config cfg;
    cfg.optimizer_type = get_json_val(cfg_content, "optimizer_type");
    if (cfg.optimizer_type.empty()) cfg.optimizer_type = "all";

    std::string s_common = get_section(cfg_content, "common");
    cfg.population_size = get_int(s_common, "population_size", 100);
    cfg.num_generations = get_int(s_common, "num_generations", 1000);
    cfg.risk_parameter  = get_dbl(s_common, "risk_parameter", 0.5);
    cfg.cardinality_K   = get_int(s_common, "cardinality_K", 10);
    cfg.w_min           = get_dbl(s_common, "w_min", 0.01);
    cfg.w_max           = get_dbl(s_common, "w_max", 0.95);
    cfg.seed            = static_cast<unsigned int>(
        get_int(s_common, "seed", 42)
    );

    std::string s_pso = get_section(cfg_content, "pso");
    // Cura (2009): α; koeficienta ω1 in ω2 ~ U[0,2] se vzorčita v pso.cpp.
    cfg.pso_alpha = get_dbl(s_pso, "alpha", 0.06);

    std::string s_sa = get_section(cfg_content, "sa");
    cfg.sa_alpha               = get_dbl(s_sa, "alpha", 0.95);
    cfg.sa_stage_length_factor = get_int(s_sa, "stage_length_factor", 2);
    cfg.sa_initial_accept_prob = get_dbl(s_sa, "initial_accept_prob", 0.8);
    cfg.sa_stop_stages         = get_int(s_sa, "stop_stages", 5);
    cfg.sa_q1                  = get_dbl(s_sa, "q1", 0.005);
    cfg.sa_q2                  = get_dbl(s_sa, "q2", 0.001);
    cfg.sa_q_switch_threshold  = get_int(s_sa, "q_switch_threshold", 10);

    PortfolioData pd;

    // Tikerji se berejo iz bridge.json, da vrstni red ustreza μ in Σ.
    std::string tickers_str = get_json_val(bridge_content, "tickers");
    if (tickers_str.empty()) {
        tickers_str = get_json_val(cfg_content, "tickers");
    }

    pd.tickers = parse_list_str(tickers_str);
    pd.n = static_cast<int>(pd.tickers.size());

    if (pd.n == 0) {
        std::cerr << "Napaka: Ni tikerjev v bridge.json ali config.json\n";
        return 1;
    }

    // μ: obvezno iz data_bridge.json.
    std::string mu_str = get_json_val(bridge_content, "mu");
    if (!mu_str.empty()) {
        pd.mu = parse_list_dbl(mu_str);
    }

    if (static_cast<int>(pd.mu.size()) != pd.n) {
        std::cerr
            << "Napaka: mu v data_bridge.json ne ustreza stevilu tikerjev ("
            << pd.mu.size() << " != " << pd.n << ")\n";
        return 1;
    }

    // Σ: obvezno iz data_bridge.json.
    std::string cov_str = get_json_val(bridge_content, "cov");
    if (cov_str.empty()) {
        std::cerr
            << "Napaka: Kovariancna matrika ni najdena v "
            << bridge_file << ".\n";
        return 1;
    }

    std::vector<double> cov_flat = parse_list_dbl(cov_str);
    if (static_cast<int>(cov_flat.size()) != pd.n * pd.n) {
        std::cerr
            << "Napaka: Pricakujem " << pd.n * pd.n
            << " elementov v \"cov\", dobil "
            << cov_flat.size() << ".\n";
        return 1;
    }

    pd.cov.assign(pd.n, std::vector<double>(pd.n));
    for (int i = 0; i < pd.n; ++i) {
        for (int j = 0; j < pd.n; ++j) {
            pd.cov[i][j] = cov_flat[i * pd.n + j];
        }
    }

    // ── Meje ε_i in δ_i ─────────────────────────────────────
    // Če niso podane za vsako sredstvo, se uporabi globalni w_min/w_max.
    std::string eps_str   = get_json_val(bridge_content, "eps");
    std::string delta_str = get_json_val(bridge_content, "delta");

    if (!eps_str.empty()) {
        cfg.eps = parse_list_dbl(eps_str);
        if (static_cast<int>(cfg.eps.size()) != pd.n) {
            cfg.eps.clear();
        }
    }

    if (!delta_str.empty()) {
        cfg.delta = parse_list_dbl(delta_str);
        if (static_cast<int>(cfg.delta.size()) != pd.n) {
            cfg.delta.clear();
        }
    }

    // ── Preveri izbiro algoritma ─────────────────────────────
    std::string algo = cfg.optimizer_type;
    std::transform(
        algo.begin(),
        algo.end(),
        algo.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );

    if (algo != "pso" && algo != "sa" && algo != "all") {
        std::cerr
            << "Napaka: Neznan optimizer_type '"
            << cfg.optimizer_type
            << "'. Dovoljene vrednosti so: pso, sa, all.\n";
        return 1;
    }

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

    // ── Vsak algoritem sam rešuje kardinalnost K ─────────────
    auto show = [&](const std::string& name,
                    const std::vector<double>& w) {
        double ret  = mv_return(w, pd);
        double risk = mv_risk(w, pd);

        print_result(
            name,
            pd.tickers,
            w,
            ret,
            risk,
            cfg.risk_parameter
        );

        std::cout
            << "  Ovrednotenj ciljne funkcije: "
            << g_eval_count << "\n";
    };

    // Pred vsakim algoritmom ponastavi števec ovrednotenj.
    auto run_algo = [&](
        const std::string& label,
        unsigned int seed_off,
        std::vector<double>(*fn)(
            const PortfolioData&,
            const Config&,
            std::mt19937&
        )
    ) {
        g_eval_count = 0;
        std::mt19937 rng(cfg.seed + seed_off);
        auto w = fn(pd, cfg, rng);
        show(label, w);
    };

    if (algo == "pso" || algo == "all") {
        std::cout << "\n[PSO] Zaganjam (Cura 2009)...\n";
        run_algo("PSO (Cura 2009)", 2, run_pso);
    }

    if (algo == "sa" || algo == "all") {
        std::cout
            << "\n[SA] Zaganjam SA (Crama & Schyns 2003)...\n";
        run_algo(
            "SA – Simulated Annealing (Crama & Schyns 2003)",
            3,
            run_sa
        );
    }

    std::cout << "\nOptimizacija zakljucena.\n";
    return 0;
}
