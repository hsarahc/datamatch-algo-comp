#include "../scoring.hpp"

// bonus function makes a meaningful contribution to score calculation

int main() {
    User u1;
    User u2;

    u1.gender = MALE;
    u2.gender = FEMALE;

    u1.loveprefs.insert(FEMALE);
    u2.loveprefs.insert(MALE);

    u1.id = 1;
    u2.id = 2;
    u1.crush = u2.crush = -1; // no crushes

    u1.year = 0;
    u2.year = 0;

    u1.min_compatible_age = 20;
    u1.max_compatible_age = 20;
    u2.min_compatible_age = 20;
    u2.max_compatible_age = 20;
    u1.age = 20;
    u2.age = 20;
    
    u1.college = "Harvard";
    u2.college = "Harvard";
    u1.house = "Adams";
    u2.house = "Quincy";
    u1.no_house_matches = false;
    u2.no_house_matches = false;

    College harvard;
    harvard.num_questions = 10;
    harvard.num_options = 5;
    harvard.exp_sim = 2.0f;
    harvard.dist_min = 0.01f;
    harvard.cosine_sims.resize(10 * 5 * 5, 0.5f);
    harvard.answer_dist.resize(10 * 5, 0.2f);

    std::unordered_map<std::string, College> college_dists;
    college_dists["Harvard"] = harvard;

    printf("Score: %f\n", calculate_score(u1, u2, 1.0f, 10, college_dists, 0.5f, 0.5f, 0.5f));
    return 0;
}