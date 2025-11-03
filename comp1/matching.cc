#include <set>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <climits>

// ================================================
// Helper functions
// ================================================

/// addmatch(a, b, matched, matched_sets):
///     Add a match between individuals `a` and `b`, which updates the `matched`
///     matrix and the `match_sets` accordingly.

static void addmatch(int a, int b, bool** matched, std::set<int>* match_sets) {
    assert(!matched[a][b] && !matched[b][a]);
    matched[a][b] = matched[b][a] = true;
    match_sets[a].insert(b);
    match_sets[b].insert(a);
}

/// removematch(a, b, matched, has, match_sets):
///     Remove the match between individuals `a` and `b`, updating the `matched`
///     structure and updating `has` and `match_sets` accordingly.

static void removematch(int a, int b, bool** matched, std::set<int>* match_sets) {
    assert(matched[a][b] && matched[b][a]);
    matched[a][b] = matched[b][a] = false;
    match_sets[a].erase(b);
    match_sets[b].erase(a);
}

/// make_sortedindices(sorted_indices, scores, ucount):
///     For each individual, create a list of indices sorted by their scores.

void make_sortedindices(int** sorted_indices, int** scores,
                               const int ucount) {
    for (int i = 0; i < ucount; ++i) {
        std::iota(sorted_indices[i], sorted_indices[i] + ucount, 0);
        // Sort candidate indices for i, descending by score[i][j].
        // If scores are equal, keep lower index first.
        std::sort(sorted_indices[i], sorted_indices[i] + ucount,
                  [&](int a, int b){
                      int sa = scores[i][a];
                      int sb = scores[i][b];
                      if (sa != sb) return sa > sb; // descending
                      return a < b;
                  });
    }
}

/// totalmatches(ucount, match_sets):
///     Calculate the total number of matches made.

static int totalmatches(const int ucount, const std::set<int>* match_sets) {
    int total = 0;
    for (int i = 0; i < ucount; ++i) {
        total += match_sets[i].size();
    }
    return total / 2; // each edge is counted twice
}

/// totalscore(scores, ucount, match_sets):
///     Calculate the total score of all matches made.

static long long totalscore(int** scores, int ucount, const std::set<int>* match_sets) {
    long long total = 0;
    for (int i = 0; i < ucount; ++i) {
        for (int j : match_sets[i]) {
            total += scores[i][j];
        }
    }
    return total / 2; // each edge is counted twice
}

/// static void output(scores, ucount, std::set<int>* match_sets):
///    Output the current matching statistics.

void output(int** scores, const int ucount, std::set<int>* match_sets) {
    std::printf("Total matches: %d\n", totalmatches(ucount, match_sets));
    std::printf("Total match score: %lld\n", totalscore(scores, ucount, match_sets));
}

/// find_worst_partner(candidate, scores, match_sets):
///    Returns the index of the worst current partner for `candidate`, according
///    to the `candidate`'s preference scores.

static int find_worst_partner(int candidate, int** scores,
                              const std::set<int>* match_sets) {
    int worst_partner = -1;
    int worst_score   = INT_MAX;
    for (int m : match_sets[candidate]) {
        int score = scores[candidate][m];
        if (score < worst_score) {
            worst_score = score;
            worst_partner = m;
        }
    }
    return worst_partner;
}


// ================================================
// TODO
//
// Note:
// You are encouraged to use std::vector and other C++ standard library
// features as needed. Do not change the function signatures. We've constructed
// the function signatures to match C-style arrays (because this is what our 
// codebase actually looks like), but you should always avoid using raw pointers
// and C-style arrays when possible.
// ================================================

/// try_propose(proposer, iter, sorted_indices, scores, matched, cur_poss,
///             change, ucount, has, match_sets):
///     A proposer advances through their preference list and attempts to match
///     with the first feasible candidate.
///
///     Rules:
///      - If a proposer has a score of -1 with a candidate 
///        (i.e. score[proposer][candidate] == -1), they cannot propose
///        to that candidate. Skip over them.
///      - If a proposer and candidate are not currently matched, and candidate
///        has the capacity to accept a new match
///        (i.e. match_sets[candidate].size() < iter), then a match is made.
///      - Else, if the candidate is at capacity, compare scores at the candidate.
///        If the proposer is strictly better than some current partner m in
///        candidate's match set, replace them using `removematch` and a 
///        subsequent `addmatch`. You may find the helper function 
///        `find_worst_partner` useful for this. 
///      - Returns true if a proposal was attempted (regardless of success), 
///        false otherwise.
///
///     Parameters:
///      - proposer: The index of the individual making the proposal.
///      - iter: The number of iterations (capacity) allowed for each individual.
///      - sorted_indices: A 2D array where each row contains the indices of
///                        candidates sorted by preference for each proposer.
///                        Example: sorted_indices[proposer] gives the
///                        preference order for `proposer` from most to least
///                        preferred.
///      - scores: A 2D array where scores[i][j] represents the score of
///                individual `i` for individual `j`. A score of -1 indicates
///                that `i` cannot propose to `j`.
///      - matched: A 2D boolean array where matched[i][j] is true
///                 if individuals `i` and `j` are currently matched.
///      - cur_poss: An array where cur_poss[i] indicates the current position
///                   in the preference list for proposer `i`.
///      - ucount: The total number of individuals.
///      - match_sets: An array of sets where match_sets[i] contains the indices
///                    of individuals currently matched with individual `i`.

bool try_propose(const int proposer, 
                        const size_t iter,
                        int** sorted_indices, 
                        int** scores, 
                        bool** matched, 
                        int* cur_poss,
                        const int ucount, 
                        std::set<int>* match_sets) {
    while (cur_poss[proposer] < ucount) {
        int candidate = sorted_indices[proposer][cur_poss[proposer]++];

        if (scores[proposer][candidate] == -1 || candidate == proposer)
            continue;
        if (!matched[proposer][candidate]) {

            if (match_sets[candidate].size() < iter) {
                addmatch(proposer, candidate, matched, match_sets);
                return true;
            } 
            else {
                int worst_partner = find_worst_partner(candidate, scores, match_sets);
                if (scores[candidate][proposer] > scores[candidate][worst_partner]) {
                    removematch(candidate, worst_partner, matched, match_sets);
                    addmatch(proposer, candidate, matched, match_sets);
                    return true;
                }
            }
        } else {
            continue;
        }
    }
    return false;
}

/// make_matches(target, sorted_indices, scores, matched, cur_poss,
///              ucount, match_sets):
///     Iterate proposal rounds from `iter = 1` to `iter = target`. In each
///     round, keep making proposal until a full pass produces no changes. 
///     Ensure that `cur_poss` is reset at the start of the function (i.e. 
///     `cur_poss[i] = 0` for all `i`). Only propose if the proposer has
///     remaining capacity (i.e. `match_sets[proposer].size() < iter`).

void make_matches(const int target, 
                  int** sorted_indices, 
                  int** scores,
                  bool** matched, 
                  int* cur_poss,
                  const int ucount,
                  std::set<int>* match_sets) {

    for (int i = 0; i < ucount; ++i) {
        cur_poss[i] = 0;
    }

    // Iterate over capacities from 1 to target
    for (int iter = 1; iter <= target; ++iter) {
        bool changed = true;

        // Continue rounds until no proposal succeeds
        while (changed) {
            changed = false;

            for (int proposer = 0; proposer < ucount; ++proposer) {
                // Only propose if proposer has remaining capacity for this iteration
                if ((int)match_sets[proposer].size() < iter) {
                    bool proposed = try_propose(proposer, iter, sorted_indices, scores, matched, cur_poss, ucount, match_sets);
                    if (proposed) {
                        changed = true;
                    }
                }
            }
        }
    return;
    }
}

// ================================================
// END OF TODO 
// ================================================

static void run(int target, int** sorted_indices,
                int** scores, bool** matched,
                int* cur_poss, int ucount,
                std::set<int>* match_sets) {
    // Clear state
    for (int i = 0; i < ucount; ++i) {
        match_sets[i].clear();
        for (int j = 0; j < ucount; ++j) matched[i][j] = false;
    }

    make_sortedindices(sorted_indices, scores, ucount);

    make_matches(target, sorted_indices, scores, matched, 
                 cur_poss, ucount, match_sets);

    // Sanity: no forbidden edges used when someone is totally isolated
    for (int i = 0; i < ucount; ++i) {
        for (int j = 0; j < ucount; ++j) {
            if (match_sets[i].empty() && match_sets[j].empty()) {
                assert(scores[i][j] < 0 && "Did you implement try_propose and make_matches yet?");
            }
        }
    }

    output(scores, ucount, match_sets);
}

void perform_matching(const int target, int** scores,
                      bool** matched, const int ucount) {
    std::vector<std::set<int>> match_sets_vec(ucount);
    std::vector<int>            cur_poss_vec(ucount, 0);
    std::vector<std::vector<int>> sorted_indices_vec(ucount, std::vector<int>(ucount));

    std::vector<int*>  sorted_indices_rows(ucount);
    for (int i = 0; i < ucount; ++i) sorted_indices_rows[i] = sorted_indices_vec[i].data();

    run(target,
        sorted_indices_rows.data(),
        scores,
        matched,
        cur_poss_vec.data(),
        ucount,
        match_sets_vec.data());
}

