// main.cc
// SLIM MAIN: no crossmatching, no crush roulette, no bios/short/spotify sims.
// Focuses on users + answer-choice similarities → scoring → matching.

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

#include "lib/json.hpp"

#include "checking.h"
#include "matching.h"
#include "scoring.h" // now provides vector-backed College
#include "user.h"
#include "utils.h"

using json = nlohmann::json;

#define SPONSORED         2
#define MAX_SPONSORED     2
#define TOTAL             10
#define MAX_TOTAL         17
#define SPONSORED_CUTOFF  1e15
#define NUM_OPTIONS       5
#define FRESHMAN_CLASS    2025

static inline Gender str2gender(const std::string s) {
    if (s == "man")      return MALE;
    if (s == "woman")    return FEMALE;
    if (s == "nonbinary")return NONBINARY;
    return UNSPECIFIED;
}

void sponsor(const size_t a, const size_t b, std::vector<long long>& boostedscores_mat, size_t tucount) {
    size_t idx1 = a * tucount + b;
    size_t idx2 = b * tucount + a;
    if (boostedscores_mat[idx1] < (long long)SPONSORED_CUTOFF) boostedscores_mat[idx1] += (long long)SPONSORED_CUTOFF;
    if (boostedscores_mat[idx2] < (long long)SPONSORED_CUTOFF) boostedscores_mat[idx2] += (long long)SPONSORED_CUTOFF;
}

bool comparator(std::pair<size_t, long long>& a, std::pair<size_t, long long>& b) {
  return a.second < b.second;
}

static void create_user_from_json(
    const std::string uid, const json& private_info, const json& public_info,
    const std::string curr_college, const json& responses, const size_t nquestions,
    std::vector<User>& users, std::vector<std::string>& uids,
    std::vector<std::string>& emails, size_t num_options
) {
    Logger logger = Logger::instance();

    if (!private_info.contains(uid) ||
        !public_info.contains(uid) ||
        !public_info[uid].contains("college") ||
        curr_college.find(public_info[uid]["college"]) == std::string::npos) {
        return;
    }

    std::string college = public_info[uid]["college"];
    User u;
    u.id = uid;
    u.college = college;
    u.email = private_info[uid]["email"];

    try {
        u.answers = responses[uid].get<std::vector<int>>();
    } catch (...) {
        u.answers = getrandanswers(nquestions, num_options);
    }
    if (u.answers.size() < nquestions) {
        u.answers = getrandanswers(nquestions, num_options);
    }

    if (private_info[uid].contains("gender") &&
        private_info[uid]["gender"].contains("genderValue")) {
        u.gender = str2gender(private_info[uid]["gender"]["genderValue"]);
    } else {
        u.gender = UNSPECIFIED;
    }

    if (private_info[uid].contains("matchCategory")) {
        u.seriousness = private_info[uid]["matchCategory"];
    }

    u.has_prompts = private_info[uid].contains("prompt");
    if (private_info[uid].contains("description")) {
        u.bio = private_info[uid]["description"];
    }

    if (public_info[uid].contains("year")) {
        std::string year = public_info[uid]["year"];
        u.year = (year == "grad" || year == "alumni") ? 5 : (FRESHMAN_CLASS - std::stoi(year));
    }
    if (public_info[uid].contains("dorm")) {
        u.house = public_info[uid]["dorm"];
    }
    if (private_info[uid].contains("noDormMatch")) {
        u.no_house_matches = private_info[uid]["noDormMatch"];
    }

    if (private_info[uid].contains("lookingForGender")) {
        std::string prefs;
        if (private_info[uid]["lookingForGender"].contains("love")) {
            prefs = private_info[uid]["lookingForGender"]["love"];
            u.loveprefs = get_gender_prefs(prefs);
        }
        if (private_info[uid]["lookingForGender"].contains("friendship")) {
            prefs = private_info[uid]["lookingForGender"]["friendship"];
            u.friendprefs = get_gender_prefs(prefs);
        }
    } else {
        u.friendprefs = get_gender_prefs("people of all genders");
    }

    users.push_back(u);
    uids.push_back(uid);
    emails.push_back(private_info[uid]["email"]);
}

int main(int argc, char** argv) {
    Logger logger = Logger::init(Logger::create_file());

    if (argc < 3) {
        logger.log(ERROR, "Not enough arguments!");
        logger.log(ERROR, "Usage: ./main FIREBASE_EXPORT.json ANSWER_SIMS_PATH");
        exit(EXIT_SUCCESS);
    }

    logger.log(INFO, "Beginning file input");
    std::ifstream userin(argv[1]);
    std::string answer_choice_sims_path = argv[2];

    if (!userin) {
        logger.log(ERROR, "File %s not found", argv[1]);
        exit(EXIT_SUCCESS);
    }
    json full;
    userin >> full;

    std::vector<std::string> required_fields = {
        "privateProfile", "publicProfile", "responses"
    };
    if (!validate_json(full, required_fields)) {
        logger.log(ERROR, "Invalid JSON!");
        exit(EXIT_SUCCESS);
    }

    std::ifstream college_ifs("college_metadata.json");
    json college_md;
    college_ifs >> college_md;
    const size_t nschools = college_md.size();

    // Use vector instead of VLA
    std::vector<std::string> colleges;
    colleges.reserve(nschools);
    for (auto& elt : college_md.items()) {
        colleges.push_back(elt.key());
    }

    // ship params per-college
    std::vector<float> ship_mus(nschools, 0.0f);
    std::vector<float> ship_sigs(nschools, 0.0f);

    std::string path = "./results";
    struct stat st;
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), (mode_t)0777);
    }

    // Map of college name -> College (from scoring.h)
    std::unordered_map<std::string, College> college_dists;

    for (size_t col = 0; col < nschools; col++) {
        logger.log(INFO, "Beginning algorithm for %s", colleges[col].c_str());

        json private_info = full.at("privateProfile");
        json public_info  = full.at("publicProfile");
        json answers      = full.at("responses");

        json cosine_similarities;
        std::string filename = answer_choice_sims_path +
            "/answer_choice_sims_" + colleges[col] + ".json";
        std::ifstream cosinein(filename);
        if (!cosinein) {
            logger.log(ERROR, "File %s not found", filename.c_str());
            exit(EXIT_SUCCESS);
        }
        cosinein >> cosine_similarities;

        json cosine_sims_list = cosine_similarities["data"];
        size_t nquestions = cosine_similarities["num_questions"];

        College college;
        college.name = colleges[col];

        // Build cosine_sims & answer_dist using vectors
        college.cosine_sims.clear();
        college.cosine_sims.resize(nquestions);
        college.answer_dist.clear();
        college.answer_dist.resize(nquestions);

        for (size_t j = 0; j < nquestions; ++j) {
            json cosine_sims_matrix = cosine_sims_list[j].at("sim_matrix");
            size_t num_options = cosine_sims_matrix[0].size();

            college.cosine_sims[j].assign(num_options, std::vector<float>(num_options));
            college.answer_dist[j].assign(num_options, 0.0f);

            for (size_t z = 0; z < num_options; ++z) {
                for (size_t x = 0; x < num_options; ++x) {
                    college.cosine_sims[j][z][x] = cosine_sims_matrix[z][x];
                }
            }
        }

        college_dists.insert({colleges[col], std::move(college)});

        logger.log(INFO, "Reading users & answers");
        std::vector<User> users;
        std::vector<std::string> uids;
        std::vector<std::string> emails;

        for (auto it = answers.begin(); it != answers.end(); ++it) {
            create_user_from_json(it.key(), private_info, public_info,
                colleges[col], answers, nquestions, users, uids, emails, 5);
        }

        const size_t tucount = users.size();
        if (!tucount) {
            logger.log(INFO, "%s has no users\n", colleges[col].c_str());
            continue;
        }

        // --- Compute answer distributions ---
        auto& ad = college_dists.at(colleges[col]).answer_dist;
        for (const User& u : users) {
            for (size_t i = 0; i < u.answers.size(); ++i) {
                size_t ans = static_cast<size_t>(u.answers[i]);
                if (i < ad.size() && ans < ad[i].size()) {
                    ++ad[i][ans];
                }
            }
        }
        for (size_t i = 0; i < nquestions; ++i) {
            size_t num_options = cosine_sims_list[i].at("sim_matrix")[0].size();
            for (size_t j = 0; j < num_options; ++j) {
                ad[i][j] /= double(tucount);
                college_dists[colleges[col]].dist_min = std::min(college_dists[colleges[col]].dist_min, ad[i][j]);
                college_dists[colleges[col]].exp_sim += std::pow(ad[i][j], 2);
            }
        }

        // --- Setup weights, scores, matchtypes (contiguous storage) ---
        std::vector<float> weights_mat(tucount * tucount, -2.0f);
        std::vector<float> scores_mat(tucount * tucount, -1.0f);
        std::vector<int>   matchtypes_mat(tucount * tucount, 0);
        std::vector<float> shipmat_mat(tucount * tucount, 0.0f);

        auto weights = [&](size_t i, size_t j)->float& { return weights_mat[i * tucount + j]; };
        auto scores  = [&](size_t i, size_t j)->float& { return scores_mat[i * tucount + j]; };
        auto matchtypes = [&](size_t i, size_t j)->int& { return matchtypes_mat[i * tucount + j]; };
        auto shipmat = [&](size_t i, size_t j)->float& { return shipmat_mat[i * tucount + j]; };

        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i; j < tucount; ++j) {
                float weight;
                if (j == i) weight = -1.f;
                else if (weights_mat[j * tucount + i] != -2.f) weight = weights_mat[i * tucount + j];
                else weight = calculate_weight(&users[i], &users[j], shipmat(i,j), ship_mus[col], ship_sigs[col]);

                weights(i,j) = weight;
                weights(j,i) = weight;

                MatchType mt = getmatchtype(&users[i], &users[j]);
                int mtv = (mt == LOVE) ? 1 : (mt == FRIENDSHIP ? 0 : -1);
                matchtypes(i,j) = mtv;
                matchtypes(j,i) = mtv;
            }
        }

        // --- Blocklist ---
        for (size_t i = 0; i < tucount; ++i) {
            User* user = &users[i];
            if (private_info[user->id].contains("blocklist")) {
                std::set<std::string> blocklist;
                for (auto& email : private_info[user->id]["blocklist"]) {
                    blocklist.insert(email);
                    size_t blocked_idx = get_index(emails, email);
                    user->blocklist = blocklist;
                    if (blocked_idx < tucount) {
                        weights(i, blocked_idx) = -1;
                        weights(blocked_idx, i) = -1;
                        matchtypes(i, blocked_idx) = -1;
                        matchtypes(blocked_idx, i) = -1;
                    }
                }
            }
        }

        /// TODO: Make noDormMatch logic here. You can make helper functions if
        /// you want.
        logger.log(INFO, "Applying noDormMatch constraints");
        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i + 1; j < tucount; ++j) {
                if ((users[i].no_house_matches || users[j].no_house_matches) &&
                    users[i].house == users[j].house &&
                    !users[i].house.empty() && !users[j].house.empty()) {
            
                    weights[i][j] = -1.f;
                    weights[j][i] = -1.f;
                    matchtypes[i][j] = -1;
                    matchtypes[j][i] = -1;
                }
            }
        }   

        logger.log(INFO, "Computing scores");
        std::unordered_map<std::string, std::string> cross_schools; // empty
        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i + 1; j < tucount; ++j) {
                if ((users[i].no_house_matches || users[j].no_house_matches) &&
                    !users[i].house.empty() && users[i].house == users[j].house) {
                    weights(i,j) = -1.f;
                    weights(j,i) = -1.f;
                    matchtypes(i,j) = -1;
                    matchtypes(j,i) = -1;
                }
            }
        }

        // --- Prepare mscores, boostedscores, matched matrices (contiguous) ---
        std::vector<long long> boostedscores_mat(tucount * tucount, 0);
        std::vector<int> mscores_mat(tucount * tucount, -1);
        std::vector<uint8_t> matched_mat(tucount * tucount, 0); // use uint8_t for storage stability

        auto boostedscores = [&](size_t i, size_t j)->long long& { return boostedscores_mat[i * tucount + j]; };
        auto mscores = [&](size_t i, size_t j)->int& { return mscores_mat[i * tucount + j]; };
        auto matched = [&](size_t i, size_t j)->uint8_t& { return matched_mat[i * tucount + j]; };

        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = 0; j < tucount; ++j) {
                int v = int(scores(i,j) * 1000000.0f);
                if (v <= 0) v = -1;
                mscores(i,j) = v;
                boostedscores(i,j) = v;
            }
        }

        std::fill(matched_mat.begin(), matched_mat.end(), 0);
    }

    logger.log(INFO, "Done!");
    return 0;
}
