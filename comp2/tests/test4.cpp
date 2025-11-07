#include "../scoring.hpp"

// check_compatibility returns true for mutual crushes

int main() {
    User u1;
    User u2;

    u1.gender = MALE;
    u2.gender = FEMALE;

    u1.loveprefs.insert(FEMALE);
    u2.loveprefs.insert(MALE);

    u1.id = 1;
    u2.id = 2;
    u1.crush = u2.id;
    u2.crush = u1.id;

    return !check_compatibility(u1, u2);
}