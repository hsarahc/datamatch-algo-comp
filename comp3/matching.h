#ifndef MATCHING_H
#define MATCHING_H

// perform_matching(target, limit, scores, matches, ucount)
//   Runs matching given the following parameters:
//   - target: The target amount of matches each user should have
//   - limit: The absolute max amount of matches any one user will have.
//   - scores: The N x N scores matrix
//   - matches: An N x N matches where `matches[i][j] == true` signifies that
//     i and j are matched
//   - ucount: N users
void perform_matching(int target, int limit, int** scores, bool** matches,
                      int ucount);

#endif
