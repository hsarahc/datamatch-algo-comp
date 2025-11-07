#include "../scoring.hpp"

// check_compatibility respects love/friendship preferences

int main() {
    User u1;
    User u2;
    User u3;
    User u4;

    u1.gender = MALE;
    u2.gender = FEMALE;
    u3.gender = MALE;
    u4.gender = MALE;

    u1.id = 1;
    u2.id = 2;
    u3.id = 3;
    u4.id = 4;

    u1.college = "Harvard";
    u2.college = "Harvard";
    u3.college = "Harvard";
    u4.college = "Harvard";

    // success
    u1.loveprefs.insert(FEMALE);
    u2.loveprefs.insert(MALE);

    // failure
    u3.friendprefs.insert(MALE);
    u4.loveprefs.insert(MALE);

    return !(check_compatibility(u1, u2) && !check_compatibility(u3, u4));
}