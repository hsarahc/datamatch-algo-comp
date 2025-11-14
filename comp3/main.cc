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
#include "lib/json.hpp"

#include "checking.h"
#include "matching.h"
#include "scoring.h"
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

// ./main data.json answer_choice_sims

/// str2gender(s)
///     Converts a string representation of gender to the Gender enum.

static inline Gender str2gender(const std::string s) {
    if (s == "man")      return MALE;
    if (s == "woman")    return FEMALE;
    if (s == "nonbinary")return NONBINARY;
    return UNSPECIFIED;
}

/// sponsor(a, b, boostedscores)
///     Boosts the score between users a and b.

void sponsor(const size_t a, const size_t b, long long** boostedscores) {
    if (boostedscores[a][b] < SPONSORED_CUTOFF) boostedscores[a][b] += SPONSORED_CUTOFF;
    if (boostedscores[b][a] < SPONSORED_CUTOFF) boostedscores[b][a] += SPONSORED_CUTOFF;
}

/// comparator(a, b)
///     Comparator for sorting pairs by second element.

bool comparator(std::pair<size_t, long long>& a,
                std::pair<size_t, long long>& b) {
  return a.second < b.second;
}

/// create_user_from_json(...)
///     Creates a User object from JSON data and appends to users, uids, emails.

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

    // Answers (fallback to random if missing/short). 
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

    // Gender prefs; default everyone for friendship.
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

/// main(argc, argv)
///    The beast. Runs all steps of the algorithm. Expects two arguments:
///    1) Firebase export JSON file path
///    2) Answer choice similarities JSON file path
///
///    In the real world, you would get the Firebase data from the Web Team 
///    director(s). Here, I've provided some for you to test with in the data/ 
///    directory.
/// 
///    In the real world, you would also pass in some additional parameters such
///    as Spotify weights, bio/short answer weights, etc. You would calculate
///    these manually via several Python scripts. Here, we skip all of that.
///    This "slim" version focuses on the core matching algorithm using only
///    answer-choice similarities.

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

    printf("Checking if college_metadata.json exists in this directory...\n");
    std::ifstream college_ifs("college_metadata.json");
    json college_md;
    college_ifs >> college_md;
    const size_t nschools = college_md.size();

    std::string colleges[nschools];
    size_t count = 0;
    for (auto& elt : college_md.items()) {
        colleges[count++] = elt.key();
    }

    // Per-school stats used by weighting.
    float ship_mus[nschools];   // kept as zeros
    float ship_sigs[nschools];  // kept as zeros
    memset(ship_mus, 0, nschools * sizeof(float));
    memset(ship_sigs, 0, nschools * sizeof(float));

    // Output dir
    std::string path = "./results";
    struct stat st;
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), (mode_t)0777);
    }

    std::unordered_map<std::string, College> college_dists;

    for (size_t col = 0; col < nschools; col++) {
        logger.log(INFO, "Beginning algorithm for %s", colleges[col].c_str());

        json private_info = full.at("privateProfile");
        json public_info  = full.at("publicProfile");
        json answers      = full.at("responses");

        // --- Load answer-choice cosine sims ---
        logger.log(INFO, "Loading answer choice sims");
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

        // Build cosine_sims & answer_dist
        float*** cosine_sims = new float**[nquestions];
        for (size_t j = 0; j < nquestions; ++j) {
            json cosine_sims_matrix = cosine_sims_list[j].at("sim_matrix");
            size_t num_options = cosine_sims_matrix[0].size();
            cosine_sims[j] = new float*[num_options];
            for (size_t z = 0; z < num_options; ++z) {
                cosine_sims[j][z] = new float[num_options];
                for (size_t x = 0; x < num_options; x++) {
                    cosine_sims[j][z][x] = cosine_sims_matrix[z][x];
                }
            }
        }

        float** answer_dist = new float*[nquestions];
        for (size_t j = 0; j < nquestions; ++j) {
            size_t num_options = cosine_sims_list[j].at("sim_matrix")[0].size();
            answer_dist[j] = new float[num_options];
            memset(answer_dist[j], 0, num_options * sizeof(float));
        }

        college.answer_dist = answer_dist;
        college.cosine_sims = cosine_sims;
        college_dists.insert({colleges[col], college});

        logger.log(INFO, "Reading users & answers");
        std::vector<User> users;
        std::vector<std::string> uids;
        std::vector<std::string> emails;

        for (auto it = answers.begin(); it != answers.end(); ++it) {
            create_user_from_json(it.key(), private_info, public_info,
                colleges[col], answers, nquestions, users, uids, emails, /*num_options*/5);
        }

        logger.log(INFO, "Found %zu users\n", users.size());
        assert(users.size() == uids.size() && uids.size() == emails.size());
        const size_t tucount = users.size();
        if (!tucount) {
            logger.log(INFO, "%s has no users\n", colleges[col].c_str());
            continue;
        }

        // Build per-question answer distributions (normalized)
        logger.log(INFO, "Answer distributions");
        {
            College& selected = college_dists.at(colleges[col]);
            float** ad = selected.answer_dist;

            for (User u : users) {
                for (size_t i = 0; i < u.answers.size(); ++i) {
                    ++ad[i][u.answers[i]];
                }
            }
            for (size_t i = 0; i < nquestions; ++i) {
                size_t num_options = cosine_sims_list[i].at("sim_matrix")[0].size();
                for (size_t j = 0; j < num_options; ++j) {
                    ad[i][j] /= double(tucount);
                    selected.dist_min = std::min(selected.dist_min, ad[i][j]);
                    selected.exp_sim += std::pow(ad[i][j], 2);
                }
            }
            selected.answer_dist = ad;
            college_dists[colleges[col]] = selected;
        }

        logger.log(INFO, "Scoring setup");

        float** weights = new float*[tucount];
        float** scores  = new float*[tucount];
        int**   matchtypes = new int*[tucount];

        for (size_t i = 0; i < tucount; ++i) {
            weights[i] = new float[tucount];
            scores[i]  = new float[tucount];
            matchtypes[i] = new int[tucount];
            for (size_t j = 0; j < tucount; ++j) {
                weights[i][j] = -2.f;
                scores[i][j]  = -1.f;
                matchtypes[i][j] = 0;
            }
        }

        std::vector<std::vector<float>> shipmat(tucount, std::vector<float>(tucount, 0.f));

        logger.log(INFO, "Computing weights & match types");
        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i; j < tucount; ++j) {
                float weight;
                if (j == i) {
                    weight = -1.f;
                } else if (weights[j][i] != -2.f) {
                    weight = weights[i][j];
                } else {
                    weight = calculate_weight(&users[i], &users[j],
                                              shipmat[i][j], /*mu*/ship_mus[col],
                                              /*sigma*/ship_sigs[col]);
                }
                weights[i][j] = weight;
                weights[j][i] = weight;

                MatchType mt = getmatchtype(&users[i], &users[j]);
                int mtv = (mt == LOVE) ? 1 : (mt == FRIENDSHIP ? 0 : -1);
                matchtypes[i][j] = mtv;
                matchtypes[j][i] = mtv;
            }
        }

        logger.log(INFO, "Processing blocklist");
        for (size_t i = 0; i < tucount; ++i) {
            User* user = &users[i];
            if (private_info[user->id].contains("blocklist")) {
                std::set<std::string> blocklist;
                for (size_t j = 0; j < private_info[user->id]["blocklist"].size(); ++j) {
                    std::string email = private_info[user->id]["blocklist"][j];
                    blocklist.insert(email);
                    size_t blocked_idx = get_index(emails, email);
                    user->blocklist = blocklist;
                    if (blocked_idx < tucount) {
                        weights[i][blocked_idx] = -1;
                        weights[blocked_idx][i] = -1;
                        matchtypes[i][blocked_idx] = -1;
                        matchtypes[blocked_idx][i] = -1;
                    }
                }
            }
        }

        /// TODO: Make noDormMatch logic here. You can make helper functions if
        /// you want.

        logger.log(INFO, "Computing scores");
        std::unordered_map<std::string, std::string> cross_schools; // empty
        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = 0; j < tucount; ++j) {
                if (weights[i][j] >= 0) {
                    float score;
                    if (j == i) {
                        score = -1.f;
                    } else if (scores[j][i] > 0) {
                        score = scores[i][j];
                    } else {
                        score = calculate_score(&users[i], &users[j],
                                                weights[i][j], nquestions,
                                                cross_schools, college_dists,
                                                /*bio*/0.0f, /*short*/0.0f,
                                                 /*spotify*/0.0f);
                    }
                    scores[i][j] = score;
                    scores[j][i] = score;
                }
            }
        }

        for (size_t i = 0; i < tucount; ++i) delete[] weights[i];
        delete[] weights;

        logger.log(INFO, "Matching");
        std::map<std::string, std::map<std::string, json>> mpairs;
        std::map<std::string, std::map<std::string, json>> mcatalog;

        std::vector<std::pair<size_t, size_t>> matches;
        std::vector<std::pair<size_t, long long>> matchesperperson[tucount];

        long long** boostedscores = new long long*[tucount];
        int** mscores = new int*[tucount];
        bool** matched = new bool*[tucount];
        for (size_t i = 0; i < tucount; ++i) {
            boostedscores[i] = new long long[tucount];
            mscores[i] = new int[tucount];
            matched[i] = new bool[tucount];
        }

        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = 0; j < tucount; ++j) {
                mscores[i][j] = int(scores[i][j] * 1000000);
                if (mscores[i][j] <= 0) mscores[i][j] = -1;
                boostedscores[i][j] = mscores[i][j];
            }
        }

        for (size_t j = 0; j < tucount; ++j) {
            memset(matched[j], 0, tucount * sizeof(bool));
        }

        // First sponsored pass
        perform_matching(SPONSORED, MAX_SPONSORED, mscores, matched, int(tucount));
        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i + 1; j < tucount; ++j) {
                if (matched[i][j]) {
                    sponsor(i, j, boostedscores);
                    mscores[i][j] = -1;
                    mscores[j][i] = -1;
                }
            }
        }

        // Remainder
        perform_matching(TOTAL - SPONSORED, MAX_TOTAL - MAX_SPONSORED,
                         mscores, matched, tucount);

        for (size_t i = 0; i < tucount; ++i) {
            for (size_t j = i + 1; j < tucount; ++j) {
                if (matched[i][j]) {
                    matches.push_back({i, j});
                    matchesperperson[i].push_back({j, boostedscores[i][j]});
                    matchesperperson[j].push_back({i, boostedscores[i][j]});
                    mscores[i][j] = -1;
                    mscores[j][i] = -1;
                }
            }
        }

        for (size_t i = 0; i < tucount; ++i) {
            delete[] mscores[i];
            delete[] matched[i];
        }
        delete[] mscores;
        delete[] matched;

        logger.log(INFO, "Checking constraints");
        check_blocklist(users, matches);
        check_gender_prefs(users, matches, matchtypes);

        // Sort each person's matches (for sponsored accounting)
        for (size_t i = 0; i < tucount; ++i) {
            std::sort(matchesperperson[i].begin(), matchesperperson[i].end(), comparator);
        }

        std::vector<long long> matchscores;
        for (auto p : matches) {
            size_t u1 = p.first, u2 = p.second;
            matchscores.push_back(boostedscores[u1][u2]);
        }
        sort(matchscores.begin(), matchscores.end());

        logger.log(INFO, "Writing results");
        for (auto p : matches) {
            size_t u1 = p.first, u2 = p.second;
            assert(u1 < u2);

            std::string couple = users[u1].id + "-" + users[u2].id;
            if (mpairs.find(couple) != mpairs.end()) continue;

            auto low = std::lower_bound(matchscores.begin(), matchscores.end(), boostedscores[u1][u2]);
            size_t ind = size_t(low - matchscores.begin());

            std::map<std::string, json> match;
            match["rating"] = (double)ind / matchscores.size() * 0.15 + 0.85;
            match["canDate"] = (boostedscores[u1][u2] > SPONSORED_CUTOFF);
            match["relationshipType"] = matchtypes[u1][u2] ? "true love" : "friendship";
            mpairs[couple] = match;

            mcatalog[users[u1].id][users[u2].id]["matched"] = false;
            mcatalog[users[u2].id][users[u1].id]["matched"] = false;
        }

        // Emit JSON
        {
            std::string mpfilename = path + "/matches_" + colleges[col] + ".json";
            FILE* outf = fopen(mpfilename.c_str(), "w");
            json mpairs_json(mpairs);
            fprintf(outf, "%s\n", mpairs_json.dump(2).c_str());
            fclose(outf);

            std::string mcfilename = path + "/matchCatalog_" + colleges[col] + ".json";
            outf = fopen(mcfilename.c_str(), "w");
            json mcatalog_json(mcatalog);
            fprintf(outf, "%s\n", mcatalog_json.dump(2).c_str());
            fclose(outf);

            printf("Wrote %s results out to %s and %s!\n",
                   colleges[col].c_str(), mpfilename.c_str(), mcfilename.c_str());
        }

        // cleanup
        for (size_t i = 0; i < tucount; ++i) {
            delete[] boostedscores[i];
        }
        delete[] boostedscores;

        for (size_t i = 0; i < tucount; ++i) {
            delete[] scores[i];
            delete[] matchtypes[i];
        }
        delete[] scores;
        delete[] matchtypes;
    }

    logger.log(INFO, "Done!");
    return 0;
}
