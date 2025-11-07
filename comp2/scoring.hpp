#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "user.hpp"

enum class Crush {
    MUTUAL,
    ONE_WAY,
    NO_CRUSH
};

enum class MatchType {
    LOVE,
    FRIENDSHIP,
    NEITHER
};

struct College {
    // Normally, these would be a triple matrix defined as so:
    // cosine_sims[question_index][answer1][answer2]:
    //      cosine similarity between answer1 and answer2 for question_index
    //
    // However, this uses a lot of memory: O(num_questions * num_options^2).
    // 
    // A more memory-efficient way is to flatten the 3D matrix into a 1D vector.
    // Let Q be the number of questions, and A be the number of answer options.

    // cosine_sims[q][a1][a2] exists at index (q * (A * A) + a1 * A + a2)
    std::vector<float> cosine_sims; // size : O(Q * A * A)
    // answer_dists[q][a] exists at index (q * A + a)
    std::vector<float> answer_dist; // size : O(Q * A)

    float exp_sim  = 0.0f;  // expected # exact matches
    float dist_min = 1e-6f; // min of answer_dist

    std::size_t num_questions = 0;
    std::size_t num_options   = 0;

    float sim(std::size_t q, std::size_t a1, std::size_t a2) const noexcept {
        const auto A = num_options;
        return cosine_sims[q * (A * A) + a1 * A + a2];
    }
    float dist(std::size_t q, std::size_t a) const noexcept {
        const auto A = num_options;
        return answer_dist[q * A + a];
    }
};

MatchType get_match_type(const User& u1, const User& u2);
Crush     get_crush_type(const User& u1, const User& u2);

bool check_compatibility(const User& u1, const User& u2);

float calculate_rice_purity(float rps1, float rps2) noexcept;

float init_weight(const User& u1, const User& u2) noexcept;

float calculate_weight(const User& u1, const User& u2,
                       float shipval, float ship_mu, float ship_sig) noexcept;

bool is_cheating(const User& u1, const User& u2, std::size_t nquestions,
                 const std::unordered_map<std::string, College>& college_dists);

float calculate_score(
    const User& u1,
    const User& u2,
    float weight,
    std::size_t num_questions,
    const std::unordered_map<std::string, College>& college_dists,
    float bio_score,
    float short_answer_score,
    float spotify_score);
