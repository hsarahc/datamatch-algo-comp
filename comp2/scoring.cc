#include "scoring.hpp"


static constexpr int YEAR_DIFF_MAX = 2; // max allowed difference in academic year

namespace {
    /// clamp_min(val, minval):
    ///     Set `val` to be at least `minval`.

    static inline float clamp_min(float val, float minval) noexcept {
        return (val < minval) ? minval : val;
    }

    /// normal_cdf(x):
    ///     Compute the CDF of the standard normal distribution at `x`.

    static inline float normal_cdf(float x) noexcept {
        static constexpr float INV_SQRT2 = 0.70710678118f; // 1 / sqrt(2)
        return 0.5f * (1.0f + std::erf(x * INV_SQRT2));
    }

    /// to_upper(sv)::
    ///     Convert a string_view to uppercase.

    static inline std::string to_upper(std::string_view sv) {
        std::string out(sv);
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        return out;
    }
}  

/// get_match_type(u1, u2):
///     Determine the MatchType between users `u1` and `u2`.

MatchType get_match_type(const User& u1, const User& u2) {
    const bool love = u1.loveprefs.count(u2.gender) &&
                      u2.loveprefs.count(u1.gender);

    const bool friendship = u1.friendprefs.count(u2.gender) &&
                            u2.friendprefs.count(u1.gender);

    return love       ? MatchType::LOVE 
         : friendship ? MatchType::FRIENDSHIP 
         : MatchType::NEITHER;
}

/// get_crush_type(u1, u2):
///     Determine the Crush status between users `u1` and `u2`.

Crush get_crush_type(const User& u1, const User& u2) {
    return (u1.crush == u2.id && u2.crush == u1.id) ? Crush::MUTUAL
         : (u1.crush == u2.id || u2.crush == u1.id) ? Crush::ONE_WAY
         :                                            Crush::NO_CRUSH;
}

/// calculate_rice_purity(rps1, rps2):
///    Calculate the rice purity score compatibility between two users.

float calculate_rice_purity(float rps1, float rps2) noexcept {
    if (rps1 < 0.0f || rps2 < 0.0f) {
        return 1.0f;
    }

    const float alpha = 1.0f / 1000.0f;
    const float beta  = std::max(rps1, rps2) / 100.0f;
    const float diff  = std::fabs(rps1 - rps2);

    const float x = std::pow(diff, 2.4f) * alpha;
    float score   = 1.3f - std::pow(x, beta);
    score = clamp_min(score, 0.0f);

    return ((score + 2.0f) / 3.3f * 1.4f + 0.6f) * 0.75f;
}

/// is_cheating(u1, u2, nquestions, college_dists):
///    Determine if users `u1` and `u2` are likely "cheating", i.e., have 
///    suspiciously similar answers.

bool is_cheating(const User& u1, const User& u2, std::size_t nquestions,
                 const std::unordered_map<std::string, College>& college_dists) {
    if (u1.college != u2.college) {
        return false;
    }

    const auto it = college_dists.find(u1.college);
    if (it == college_dists.end()) {
        return false;
    }

    const College& college = it->second;

    std::size_t matches = 0;
    for (std::size_t i = 0; i < nquestions && i < u1.answers.size() && i < u2.answers.size(); ++i) {
        if (u1.answers[i] == u2.answers[i]) {
            matches += (u1.answers[i] == u2.answers[i]); // bool evaluates to 1 if true
        }
    }

    const bool mutual = (get_crush_type(u1, u2) == Crush::MUTUAL);
    const float threshold = mutual ? 6.5f : 8.5f;
    return (static_cast<float>(matches) - college.exp_sim) > threshold;
}

/// ================================
/// TODO
/// ================================ 

/// check_compatibility(u1, u2):
///     Check if users `u1` and `u2` are compatible matches based on various
///     criteria specified in the README.

bool check_compatibility(const User& u1, const User& u2) {
    // TODO
    return false;
}

/// bonus(u1, u2):
///    Implement this according to the README instructions.

float bonus(const User& u1, const User& u2) {
    // TODO
    return 0.0f;
}

/// ================================
/// END OF TODO
/// ================================


/// calculate_score(u1, u2, weight, num_questions, college_dists, bio_score,
///                 short_answer_score, spotify_score):
///     Calculate the compatibility score between two users.

float calculate_score(
    const User& u1,
    const User& u2,
    float weight,
    std::size_t num_questions,
    const std::unordered_map<std::string, College>& college_dists,
    float bio_score,
    float short_answer_score,
    float spotify_score
) {
    if (is_cheating(u1, u2, num_questions, college_dists)) {
        return -1.0f;
    }

    if (u1.college != u2.college) {
        return -1.0f;
    }

    const auto it = college_dists.find(u1.college);
    if (it == college_dists.end()) {
        return -1.0f; // unknown college
    }

    const College& college = it->second;
    const std::size_t nquestions = std::min({
        num_questions, college.num_questions, 
        u1.answers.size(), u2.answers.size()
    });
    const std::size_t num_options = college.num_options;

    float ml_score   = 0.0f;
    float prop_score = 0.0f;

    const float denom_guard = clamp_min(std::sqrt(college.dist_min), 1e-6f);
    const float norm_ref    = 1.0f / denom_guard;

    for (std::size_t i = 0; i < nquestions; ++i) {
        const int a1 = u1.answers[i];
        const int a2 = u2.answers[i];

        if (a1 < 0 || a1 >= static_cast<std::size_t>(num_options) ||
            a2 < 0 || a2 >= static_cast<std::size_t>(num_options)) {
            continue;
        }

        const float p1 = college.dist(i, static_cast<std::size_t>(a1));
        if (a1 == a2) {
            const float imp_wgt = 1.0f / (1.0f + p1);
            prop_score += (imp_wgt / clamp_min(std::sqrt(p1), 1e-6f)) / norm_ref;
            ml_score   += imp_wgt;
        } else {
            const float p2 = college.dist(i, static_cast<std::size_t>(a2));
            const float imp_wgt = 1.0f / (1.0f + std::sqrt(clamp_min(p1 * p2, 0.0f)));
            ml_score += imp_wgt * college.sim(i, static_cast<std::size_t>(a1), static_cast<std::size_t>(a2));
        }
    }

    float sim_score = ml_score + prop_score + bio_score * 0.10f + 
                      short_answer_score * 0.10f + spotify_score * 0.30f;

    // Your bonus will go here!
    sim_score += bonus(u1, u2);

    sim_score = clamp_min(sim_score, 0.001f);
    return weight * sim_score;
}