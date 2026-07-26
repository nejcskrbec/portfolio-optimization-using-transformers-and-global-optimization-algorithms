#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Minimalen JSON parser
static std::string get_json_val(const std::string& content,
                                const std::string& key) {
    size_t search_from = 0;
    while (true) {
        size_t kp = content.find("\"" + key + "\"", search_from);
        if (kp == std::string::npos) return "";
        size_t cp = content.find(":", kp);
        size_t st = content.find_first_not_of(" \t\n\r", cp + 1);
        if (st == std::string::npos) return "";
        if (content[st] == '[') {
            int depth = 0; size_t en = st;
            while (en < content.size()) {
                if (content[en] == '[') depth++;
                else if (content[en] == ']') { depth--; if (depth == 0) break; }
                en++;
            }
            return content.substr(st + 1, en - st - 1);
        }
        if (content[st] == '{') { search_from = st + 1; continue; }
        size_t en = content.find_first_of(",}\n", st);
        std::string v = content.substr(st, en - st);
        v.erase(std::remove(v.begin(), v.end(), '\"'), v.end());
        v.erase(0, v.find_first_not_of(" \t\n\r"));
        auto last = v.find_last_not_of(" \t\n\r");
        if (last != std::string::npos) v.erase(last + 1);
        return v;
    }
}

static std::string get_section(const std::string& content,
                                const std::string& key) {
    size_t kp = content.find("\"" + key + "\"");
    if (kp == std::string::npos) return "";
    size_t cp = content.find(":", kp);
    size_t st = content.find_first_not_of(" \t\n\r", cp + 1);
    if (st == std::string::npos || content[st] != '{') return "";
    int depth = 0; size_t en = st;
    while (en < content.size()) {
        if (content[en] == '{') depth++;
        else if (content[en] == '}') { depth--; if (depth == 0) break; }
        en++;
    }
    return content.substr(st + 1, en - st - 1);
}

static std::vector<std::string> parse_list_str(const std::string& s) {
    std::stringstream ss(s); std::string item;
    std::vector<std::string> res;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove(item.begin(), item.end(), '\"'), item.end());
        item.erase(std::remove(item.begin(), item.end(), ' '), item.end());
        item.erase(std::remove(item.begin(), item.end(), '\n'), item.end());
        item.erase(std::remove(item.begin(), item.end(), '\r'), item.end());
        if (!item.empty()) res.push_back(item);
    }
    return res;
}

static std::vector<double> parse_list_dbl(const std::string& s) {
    std::vector<std::string> sv = parse_list_str(s);
    std::vector<double> res;
    for (auto& x : sv) if (!x.empty()) res.push_back(std::stod(x));
    return res;
}

static double get_dbl(const std::string& content, const std::string& key,
                      double def = 0.0) {
    std::string v = get_json_val(content, key);
    if (v.empty()) return def;
    try { return std::stod(v); } catch (...) { return def; }
}
static int get_int(const std::string& content, const std::string& key,
                   int def = 0) {
    std::string v = get_json_val(content, key);
    if (v.empty()) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

// Struktura s podatki o portfelju
struct PortfolioData {
    int n;
    std::vector<std::string> tickers;
    std::vector<double> mu;
    std::vector<std::vector<double>> cov;
};

// Konfiguracijski parametri
struct Config {
    std::string optimizer_type;
    int population_size = 100;
    int num_generations = 1000;
    double risk_parameter = 0.5;
    int cardinality_K = 10;
    double w_min = 0.01;   // globalen εᵢ fallback (Chang et al. 2000)
    double w_max = 0.95;   // globalen δᵢ fallback

    // Per-asset εᵢ in δᵢ (Chang et al. 2000, Eq. 12: εᵢzᵢ ≤ wᵢ ≤ δᵢzᵢ)
    // Če sta prazna, se uporabi globalen w_min/w_max za vse assete.
    std::vector<double> eps;   // εᵢ: minimalna proporcija če asset v portfelju
    std::vector<double> delta; // δᵢ: maksimalna proporcija če asset v portfelju

    // PSO sledi Cura (2009): hitrostni koeficienti ω₁,ω₂ ~ U[0,2] se
    // vzorčijo znotraj pso.cpp. Konstrikcijski parametri (omega/c1/c2) NISO
    // del Cura sheme — zato jih tu ne hranimo. Ostane samo Cura-jev α.
    double pso_alpha = 0.06;

    double sa_alpha               = 0.95;
    int    sa_stage_length_factor = 2;
    double sa_initial_accept_prob = 0.8;
    int    sa_stop_stages         = 5;
    double sa_q1                  = 0.005;
    double sa_q2                  = 0.001;
    int    sa_q_switch_threshold  = 10;

    double ga_mutation_rate   = 0.01;
    double ga_mutation_sigma  = 0.05;

    unsigned int seed = 42;
};

// Števec ovrednotenj ciljne funkcije: vsak algoritem ga inkrementira ob vsakem
// ovrednotenju, main.cpp ga resetira pred vsakim algoritmom in izpiše skupno
// število → transparentna primerjava računske porabe.
static long long g_eval_count = 0;

// Pomožne matematične funkcije

static double mv_return(const std::vector<double>& w, const PortfolioData& pd) {
    double r = 0.0;
    for (int i = 0; i < pd.n; i++) r += w[i] * pd.mu[i];
    return r;
}

static double mv_risk(const std::vector<double>& w, const PortfolioData& pd) {
    // Redko-zasedena kvadratna oblika: pri kardinalnosti K≪N je le K uteži
    // neničelnih, ostali členi so 0. Iteriramo SAMO čez neničelne indekse →
    // O(N + K²) namesto O(N²). Matematično identično polni obliki.
    std::vector<int> sel;
    for (int i = 0; i < pd.n; i++) if (w[i] != 0.0) sel.push_back(i);
    double risk = 0.0;
    for (int a : sel) {
        double wa = w[a];
        const std::vector<double>& ca = pd.cov[a];
        for (int b : sel) risk += wa * w[b] * ca[b];
    }
    return risk;
}

// Mean-variance kriterij (minimizacija): -(P·μᵀw - (1-P)·wᵀΣw).
// Markowitzeva teorija portfelja (kompromis pričakovani donos ↔ varianca);
// P ∈ [0,1] je parameter nagnjenosti k tveganju (P→1 donos, P→0 varianca):
//   Markowitz, H. (1952). "Portfolio Selection." Journal of Finance
//   7(1):77–91.
static double mv_objective(const std::vector<double>& w,
                           const PortfolioData& pd, double P) {
    g_eval_count++;
    return -(P * mv_return(w, pd) - (1.0 - P) * mv_risk(w, pd));
}

// Normalizacija deležev po Chang et al. (2000) Algorithm 1:
//   Σ wᵢ = 1,  εᵢzᵢ ≤ wᵢ ≤ δᵢzᵢ  (Eq. 12)
// Podpira globalen wmin/wmax ALI per-asset eps/delta.
static void normalize_weights(std::vector<double>& w,
                               const std::vector<int>& selected,
                               double wmin, double wmax,
                               const std::vector<double>& eps   = {},
                               const std::vector<double>& delta = {}) {
    if (selected.empty()) return;
    // Per-asset meje: fallback na globalni wmin/wmax
    auto ei = [&](int i) { return (!eps.empty()   && i<(int)eps.size())   ? eps[i]   : wmin; };
    auto di = [&](int i) { return (!delta.empty() && i<(int)delta.size()) ? delta[i] : wmax; };

    for (int i : selected)
        w[i] = std::max(ei(i), std::min(di(i), w[i]));
    double sum = 0.0;
    for (int i : selected) sum += w[i];
    if (sum <= 0.0) {
        double eq = 1.0 / selected.size();
        for (int i : selected) w[i] = eq;
        return;
    }
    for (int i : selected) w[i] /= sum;
    // Iterativni postopek (Chang Algorithm 1): zadovolji εᵢ in δᵢ
    for (int iter = 0; iter < 50; iter++) {
        bool ok = true;
        double excess_hi = 0.0, excess_lo = 0.0;
        for (int i : selected) {
            if (w[i] > di(i)) { excess_hi += w[i] - di(i); w[i] = di(i); ok = false; }
            if (w[i] < ei(i)) { excess_lo += ei(i) - w[i]; w[i] = ei(i); ok = false; }
        }
        if (ok) break;
        if (excess_hi > 0.0) {
            double free = 0.0;
            for (int i : selected) if (w[i] < di(i)) free += di(i) - w[i];
            if (free > 1e-12)
                for (int i : selected)
                    if (w[i] < di(i)) w[i] += (di(i) - w[i]) / free * excess_hi;
        }
        if (excess_lo > 0.0) {
            double free = 0.0;
            for (int i : selected) if (w[i] > ei(i)) free += w[i] - ei(i);
            if (free > 1e-12)
                for (int i : selected)
                    if (w[i] > ei(i)) w[i] -= (w[i] - ei(i)) / free * excess_lo;
        }
        sum = 0.0;
        for (int i : selected) sum += w[i];
        if (sum > 0.0) for (int i : selected) w[i] /= sum;
    }
}

static void print_result(const std::string& algo,
                         const std::vector<std::string>& tickers,
                         const std::vector<double>& weights,
                         double ret, double risk, double P) {
    std::cout << "\n══════════════════════════════════════════════════════\n";
    std::cout << "  Algoritem : " << algo << "\n";
    std::cout << "  P (tveganje): " << P << "\n";
    std::cout << "  Pričakovan donos: " << std::fixed << std::setprecision(6)
              << ret * 100.0 << " %\n";
    std::cout << "  Tveganje (var): " << risk << "\n";
    std::cout << "  Portfelj:\n";
    for (int i = 0; i < (int)weights.size(); i++)
        if (weights[i] > 1e-5)
            printf("    %-6s  %8.4f %%\n", tickers[i].c_str(), weights[i] * 100.0);
    std::cout << "══════════════════════════════════════════════════════\n";
}