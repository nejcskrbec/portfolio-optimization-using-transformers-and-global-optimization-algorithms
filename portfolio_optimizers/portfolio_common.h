#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Minimalni razčlenjevalnik JSON
// -----------------------------------------------------------------------------

static std::string get_json_val(
    const std::string& content,
    const std::string& key
) {
    size_t search_from = 0;

    while (true) {
        size_t kp = content.find("\"" + key + "\"", search_from);
        if (kp == std::string::npos) return "";

        size_t cp = content.find(":", kp);
        if (cp == std::string::npos) return "";

        size_t st = content.find_first_not_of(" \t\n\r", cp + 1);
        if (st == std::string::npos) return "";

        if (content[st] == '[') {
            int depth = 0;
            size_t en = st;

            while (en < content.size()) {
                if (content[en] == '[') {
                    ++depth;
                } else if (content[en] == ']') {
                    --depth;
                    if (depth == 0) break;
                }
                ++en;
            }

            return content.substr(st + 1, en - st - 1);
        }

        if (content[st] == '{') {
            search_from = st + 1;
            continue;
        }

        size_t en = content.find_first_of(",}\n", st);
        std::string v = content.substr(st, en - st);

        v.erase(std::remove(v.begin(), v.end(), '\"'), v.end());

        size_t first = v.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return "";
        v.erase(0, first);

        size_t last = v.find_last_not_of(" \t\n\r");
        if (last != std::string::npos) {
            v.erase(last + 1);
        }

        return v;
    }
}

static std::string get_section(
    const std::string& content,
    const std::string& key
) {
    size_t kp = content.find("\"" + key + "\"");
    if (kp == std::string::npos) return "";

    size_t cp = content.find(":", kp);
    if (cp == std::string::npos) return "";

    size_t st = content.find_first_not_of(" \t\n\r", cp + 1);
    if (st == std::string::npos || content[st] != '{') return "";

    int depth = 0;
    size_t en = st;

    while (en < content.size()) {
        if (content[en] == '{') {
            ++depth;
        } else if (content[en] == '}') {
            --depth;
            if (depth == 0) break;
        }
        ++en;
    }

    return content.substr(st + 1, en - st - 1);
}

static std::vector<std::string> parse_list_str(
    const std::string& s
) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> res;

    while (std::getline(ss, item, ',')) {
        item.erase(
            std::remove(item.begin(), item.end(), '\"'),
            item.end()
        );
        item.erase(
            std::remove(item.begin(), item.end(), ' '),
            item.end()
        );
        item.erase(
            std::remove(item.begin(), item.end(), '\n'),
            item.end()
        );
        item.erase(
            std::remove(item.begin(), item.end(), '\r'),
            item.end()
        );

        if (!item.empty()) {
            res.push_back(item);
        }
    }

    return res;
}

static std::vector<double> parse_list_dbl(
    const std::string& s
) {
    std::vector<std::string> sv = parse_list_str(s);
    std::vector<double> res;

    for (const auto& x : sv) {
        if (!x.empty()) {
            res.push_back(std::stod(x));
        }
    }

    return res;
}

static double get_dbl(
    const std::string& content,
    const std::string& key,
    double def = 0.0
) {
    std::string v = get_json_val(content, key);
    if (v.empty()) return def;

    try {
        return std::stod(v);
    } catch (...) {
        return def;
    }
}

static int get_int(
    const std::string& content,
    const std::string& key,
    int def = 0
) {
    std::string v = get_json_val(content, key);
    if (v.empty()) return def;

    try {
        return std::stoi(v);
    } catch (...) {
        return def;
    }
}

// -----------------------------------------------------------------------------
// Podatki o portfelju
// -----------------------------------------------------------------------------

struct PortfolioData {
    int n = 0;
    std::vector<std::string> tickers;
    std::vector<double> mu;
    std::vector<std::vector<double>> cov;
};

// -----------------------------------------------------------------------------
// Konfiguracija optimizatorjev
// -----------------------------------------------------------------------------

struct Config {
    std::string optimizer_type;

    int population_size = 100;
    int num_generations = 1000;

    double risk_parameter = 0.5;
    int cardinality_K = 10;

    // Globalni meji kot rezervna možnost.
    double w_min = 0.01;
    double w_max = 0.95;

    // Meje po posameznem sredstvu:
    // ε_i z_i <= w_i <= δ_i z_i.
    std::vector<double> eps;
    std::vector<double> delta;

    // PSO po Curi (2009):
    // koeficienta ω1 in ω2 ~ U[0,2] se vzorčita v pso.cpp.
    double pso_alpha = 0.06;

    // Simulirano ohlajanje.
    double sa_alpha = 0.95;
    int sa_stage_length_factor = 2;
    double sa_initial_accept_prob = 0.8;
    int sa_stop_stages = 5;
    double sa_q1 = 0.005;
    double sa_q2 = 0.001;
    int sa_q_switch_threshold = 10;

    unsigned int seed = 42;
};

// -----------------------------------------------------------------------------
// Števec ovrednotenj ciljne funkcije
// -----------------------------------------------------------------------------

static long long g_eval_count = 0;

// -----------------------------------------------------------------------------
// Matematične funkcije
// -----------------------------------------------------------------------------

static double mv_return(
    const std::vector<double>& w,
    const PortfolioData& pd
) {
    double r = 0.0;

    for (int i = 0; i < pd.n; ++i) {
        r += w[i] * pd.mu[i];
    }

    return r;
}

static double mv_risk(
    const std::vector<double>& w,
    const PortfolioData& pd
) {
    // Pri kardinalnosti K << N je neničelnih le K uteži.
    std::vector<int> selected;
    selected.reserve(pd.n);

    for (int i = 0; i < pd.n; ++i) {
        if (w[i] != 0.0) {
            selected.push_back(i);
        }
    }

    double risk = 0.0;

    for (int a : selected) {
        const double wa = w[a];
        const std::vector<double>& cov_a = pd.cov[a];

        for (int b : selected) {
            risk += wa * w[b] * cov_a[b];
        }
    }

    return risk;
}

// Povprečje--varianca, zapisana kot minimizacijski problem:
//
//   g(w) = (1-P) w^T Σ w - P μ^T w
//
// kar je ekvivalentno maksimiranju:
//
//   P μ^T w - (1-P) w^T Σ w.
static double mv_objective(
    const std::vector<double>& w,
    const PortfolioData& pd,
    double P
) {
    ++g_eval_count;

    return
        (1.0 - P) * mv_risk(w, pd)
        - P * mv_return(w, pd);
}

// -----------------------------------------------------------------------------
// Projekcija uteži na dopustno množico
// -----------------------------------------------------------------------------

static void normalize_weights(
    std::vector<double>& w,
    const std::vector<int>& selected,
    double wmin,
    double wmax,
    const std::vector<double>& eps = {},
    const std::vector<double>& delta = {}
) {
    if (selected.empty()) return;

    auto lower = [&](int i) {
        return (!eps.empty() && i < static_cast<int>(eps.size()))
            ? eps[i]
            : wmin;
    };

    auto upper = [&](int i) {
        return (!delta.empty() && i < static_cast<int>(delta.size()))
            ? delta[i]
            : wmax;
    };

    // Osnovno omejevanje.
    for (int i : selected) {
        w[i] = std::max(lower(i), std::min(upper(i), w[i]));
    }

    double sum = 0.0;
    for (int i : selected) {
        sum += w[i];
    }

    if (sum <= 0.0) {
        const double equal_weight =
            1.0 / static_cast<double>(selected.size());

        for (int i : selected) {
            w[i] = equal_weight;
        }
    } else {
        for (int i : selected) {
            w[i] /= sum;
        }
    }

    // Iterativno popravljanje spodnjih in zgornjih meja.
    for (int iter = 0; iter < 50; ++iter) {
        bool feasible = true;
        double excess_high = 0.0;
        double deficit_low = 0.0;

        for (int i : selected) {
            if (w[i] > upper(i)) {
                excess_high += w[i] - upper(i);
                w[i] = upper(i);
                feasible = false;
            }

            if (w[i] < lower(i)) {
                deficit_low += lower(i) - w[i];
                w[i] = lower(i);
                feasible = false;
            }
        }

        if (feasible) {
            break;
        }

        if (excess_high > 0.0) {
            double capacity = 0.0;

            for (int i : selected) {
                if (w[i] < upper(i)) {
                    capacity += upper(i) - w[i];
                }
            }

            if (capacity > 1e-12) {
                for (int i : selected) {
                    if (w[i] < upper(i)) {
                        w[i] +=
                            (upper(i) - w[i])
                            / capacity
                            * excess_high;
                    }
                }
            }
        }

        if (deficit_low > 0.0) {
            double removable = 0.0;

            for (int i : selected) {
                if (w[i] > lower(i)) {
                    removable += w[i] - lower(i);
                }
            }

            if (removable > 1e-12) {
                for (int i : selected) {
                    if (w[i] > lower(i)) {
                        w[i] -=
                            (w[i] - lower(i))
                            / removable
                            * deficit_low;
                    }
                }
            }
        }

        sum = 0.0;
        for (int i : selected) {
            sum += w[i];
        }

        if (sum > 0.0) {
            for (int i : selected) {
                w[i] /= sum;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Izpis rezultata
// -----------------------------------------------------------------------------

static void print_result(
    const std::string& algo,
    const std::vector<std::string>& tickers,
    const std::vector<double>& weights,
    double ret,
    double risk,
    double P
) {
    std::cout
        << "\n══════════════════════════════════════════════════════\n";

    std::cout << "  Algoritem : " << algo << "\n";
    std::cout << "  P (tveganje): " << P << "\n";

    std::cout
        << "  Pričakovan donos: "
        << std::fixed
        << std::setprecision(6)
        << ret * 100.0
        << " %\n";

    std::cout << "  Tveganje (var): " << risk << "\n";
    std::cout << "  Portfelj:\n";

    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
        if (weights[i] > 1e-5) {
            std::printf(
                "    %-6s  %8.4f %%\n",
                tickers[i].c_str(),
                weights[i] * 100.0
            );
        }
    }

    std::cout
        << "══════════════════════════════════════════════════════\n";
}
