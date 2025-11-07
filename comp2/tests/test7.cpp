#include "../scoring.hpp"

// check_compatibility respects same-house preferences

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
    u2.year = 0;
    u3.year = 0;
    u4.year = 0;

    u1.min_compatible_age = 20;
    u1.max_compatible_age = 20;
    u2.min_compatible_age = 20;
    u2.max_compatible_age = 20;
    u3.min_compatible_age = 20;
    u3.max_compatible_age = 20;
    u4.min_compatible_age = 20;
    u4.max_compatible_age = 20;
    u1.age = 20;
    u2.age = 20;
    u3.age = 20;
    u4.age = 20;

    // success
    u1.college = "Harvard";
    u2.college = "Harvard";
    u1.house = "Adams";
    u2.house = "Quincy";
    u1.no_house_matches = false;
    u2.no_house_matches = false;

    // failure
    u3.college = "Harvard";
    u4.college = "Harvard";
    u3.house = "Adams";
    u4.house = "Adams";
    u3.no_house_matches = false;
    u4.no_house_matches = true;

    return !(check_compatibility(u1, u2) && !check_compatibility(u3, u4));
}