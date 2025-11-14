#ifndef SCORING_H
#define SCORING_H

#include "user.h"

#define MBTI_BEST   1.15
#define MBTI_GOOD   1.1
#define MBTI_OK     1
#define MBTI_BAD    0.95
#define MBTI_WORST  0.9

struct College {
    std::string name;
    float*** cosine_sims;
    float** answer_dist;
    float exp_sim = 0.0;
    float dist_min = std::numeric_limits<float>::max();
};

enum Crush {
    MUTUAL,
    ONE_WAY,
    NO_CRUSH,
};

enum MatchType {
    LOVE,
    FRIENDSHIP,
    NEITHER,
};

std::vector<int> getrandanswers(size_t num_questions, size_t num_options);

MatchType getmatchtype(const User* u1, const User* u2);

float calculate_weight(const User* u1, const User* u2, const float shipval,
                       const float ship_mu, const float ship_sig);

float calculate_cross_weight(const User* u1, const User* u2,
                             const float shipval);

float calculate_score(
    const User* u1,
    const User* u2,
    const float weight,
    const size_t num_questions,
    const std::unordered_map<std::string, std::string>& cross_schools,
    const std::unordered_map<std::string, College>& college_dists,
    const float bio_score,
    const float short_answer_score,
    const float spotify_score
);

float calculate_cross_score(const User* u1, const User* u2, const float weight);

float calculate_rice_purity(const float rps1, const float rps2);

float mbti_compatability(std::string mbti1, std::string mbti2);


#endif
