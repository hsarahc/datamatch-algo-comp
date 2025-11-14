// Minimal stub implementations to allow comp3/main.cc to compile and run
#include "utils.h"
#include "scoring.h"
#include "matching.h"
#include "checking.h"
#include <cstdio>
#include <cstdarg>
#include <algorithm>
#include <cmath>

// Logger static instance
Logger Logger::logger;

Logger& Logger::init(FILE* out) {
    logger.out = out ? out : stdout;
    return logger;
}

Logger& Logger::instance() {
    return logger;
}

void Logger::log(LogLevel level, const char* format, ...) {
    if (!logger.out) logger.out = stdout;
    va_list args;
    va_start(args, format);
    vfprintf(logger.out, format, args);
    fprintf(logger.out, "\n");
    va_end(args);
}

FILE* Logger::create_file() {
    return stdout;
}

// utils helpers
size_t get_index(const std::vector<std::string> v, const std::string target) {
    for (size_t i = 0; i < v.size(); ++i) if (v[i] == target) return i;
    return v.size();
}

float normal_cdf(float x) {
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

bool validate_json(const json f, const std::vector<std::string>& required_fields) {
    for (auto &k : required_fields) if (!f.contains(k)) return false;
    return true;
}

void sort_on_other(int* main, int* other, int size) {
    // simple bubble-ish using vector
    std::vector<std::pair<int,int>> v(size);
    for (int i = 0; i < size; ++i) v[i] = {other[i], main[i]};
    std::sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.first < b.first; });
    for (int i = 0; i < size; ++i) { other[i] = v[i].first; main[i] = v[i].second; }
}

std::vector<std::string> split(std::string str, const char delim) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (true) {
        size_t p = str.find(delim, pos);
        if (p == std::string::npos) { out.push_back(str.substr(pos)); break; }
        out.push_back(str.substr(pos, p-pos));
        pos = p+1;
    }
    return out;
}

// scoring stubs
std::vector<int> getrandanswers(size_t num_questions, size_t num_options) {
    std::vector<int> a(num_questions, 0);
    for (size_t i = 0; i < num_questions; ++i) a[i] = int(i % num_options);
    return a;
}

MatchType getmatchtype(const User* u1, const User* u2) {
    // simple heuristic: if either seriousness high, LOVE, else FRIENDSHIP
    if (!u1 || !u2) return NEITHER;
    if (u1->seriousness >= 2 && u2->seriousness >= 2) return LOVE;
    return FRIENDSHIP;
}

float calculate_weight(const User* u1, const User* u2, const float shipval,
                       const float ship_mu, const float ship_sig) {
    // trivial: if same house penalize
    if (!u1 || !u2) return -1.f;
    if (!u1->blocked_houses.empty() && std::find(u1->blocked_houses.begin(), u1->blocked_houses.end(), u2->house) != u1->blocked_houses.end()) return -1.f;
    return 1.0f;
}

float calculate_cross_weight(const User* u1, const User* u2, const float shipval) {
    return calculate_weight(u1, u2, shipval, 0.f, 1.f);
}

float calculate_score(const User* u1, const User* u2, const float weight,
                      const size_t num_questions,
                      const std::unordered_map<std::string, std::string>& /*cross_schools*/,
                      const std::unordered_map<std::string, College>& /*college_dists*/,
                      const float /*bio*/, const float /*short_answer*/, const float /*spotify*/) {
    if (weight < 0) return -1.f;
    // simple score: weight * number of matching answer choices
    size_t matches = 0;
    for (size_t i = 0; u1 && u2 && i < u1->answers.size() && i < u2->answers.size() && i < num_questions; ++i) {
        if (u1->answers[i] == u2->answers[i]) ++matches;
    }
    return weight * float(matches);
}

float calculate_cross_score(const User* u1, const User* u2, const float weight) {
    return calculate_score(u1, u2, weight, 0, {}, {}, 0.f,0.f,0.f);
}

float calculate_rice_purity(const float rps1, const float rps2) { return (rps1 + rps2) * 0.5f; }

float mbti_compatability(std::string mbti1, std::string mbti2) { return 1.0f; }

std::set<Gender> get_gender_prefs(const std::string prefs_str) {
    std::set<Gender> s;
    // very permissive: include all
    s.insert(MALE); s.insert(FEMALE); s.insert(NONBINARY);
    return s;
}

// matching: greedy top-scoring pairs respecting limits
void perform_matching(int target, int limit, int** scores, bool** matches, int ucount) {
    struct Entry { int score; int i; int j; };
    std::vector<Entry> v;
    for (int i = 0; i < ucount; ++i) {
        for (int j = i+1; j < ucount; ++j) {
            if (scores[i][j] > 0) v.push_back({scores[i][j], i, j});
        }
    }
    std::sort(v.begin(), v.end(), [](const Entry&a,const Entry&b){return a.score>b.score;});
    std::vector<int> counts(ucount, 0);
    for (auto &e : v) {
        if (counts[e.i] >= limit || counts[e.j] >= limit) continue;
        if (matches[e.i][e.j]) continue;
        // add match
        matches[e.i][e.j] = matches[e.j][e.i] = true;
        counts[e.i]++;
        counts[e.j]++;
    }
}

// checking
void check_blocklist(std::vector<User>& users, std::vector<std::pair<size_t,size_t>>& matches) {
    std::vector<std::pair<size_t,size_t>> kept;
    for (auto &p : matches) {
        const User &a = users[p.first];
        const User &b = users[p.second];
        if (a.blocklist.count(b.email) || b.blocklist.count(a.email)) continue;
        kept.push_back(p);
    }
    matches.swap(kept);
}

void check_gender_prefs(std::vector<User>& users, std::vector<std::pair<size_t,size_t>>& matches, int** matchtypes) {
    std::vector<std::pair<size_t,size_t>> kept;
    for (auto &p : matches) {
        const User &a = users[p.first];
        const User &b = users[p.second];
        bool ok_a = a.loveprefs.empty() || a.loveprefs.count(b.gender);
        bool ok_b = b.loveprefs.empty() || b.loveprefs.count(a.gender);
        // if matchtype == FRIENDSHIP, allow friendprefs
        if (matchtypes[p.first][p.second] == 0) {
            ok_a = a.friendprefs.empty() || a.friendprefs.count(b.gender);
            ok_b = b.friendprefs.empty() || b.friendprefs.count(a.gender);
        }
        if (ok_a && ok_b) kept.push_back(p);
    }
    matches.swap(kept);
}
