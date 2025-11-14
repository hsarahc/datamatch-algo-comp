#ifndef CHECKING_H
#define CHECKING_H

#include "user.h"

// check_blocklist(users, matches, user_idxs)
//   Iterates through all users and ensures that none of them are matched with
//   anyone on their blocklist (bidirectional).
void check_blocklist(std::vector<User>& users,
                     std::vector<std::pair<size_t, size_t>>& matches);

// check_gender_prefs(users, matches)
//   Iterates through all users and matches and ensures that all user
//   preferences have been bidirectionally honored.
void check_gender_prefs(std::vector<User>& users,
                        std::vector<std::pair<size_t, size_t>>& matches,
                        int** matchtypes);

#endif
