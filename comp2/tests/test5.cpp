#include "../scoring.hpp"

// check_compatibility respects class year gaps

int main() {
    User u1;
    User u2;
    User u3;
    User u4;

    u1.gender = MALE;
    u2.gender = FEMALE;
    u3.gender = MALE;
    u4.gender = MALE;

    u1.loveprefs.insert(FEMALE);
    u2.loveprefs.insert(MALE);
    u3.loveprefs.insert(MALE);
    u4.loveprefs.insert(MALE);

    u1.id = 1;
    u2.id = 2;
    u3.id = 3;
    u4.id = 4;
    u1.crush = u2.crush = u3.crush = u4.crush = -1; // no crushes

    u1.year = 0;
    u2.year = 1;

    u3.year = 0;
    u4.year = 3;

    return !(check_compatibility(u1, u2) && !check_compatibility(u3, u4));
}