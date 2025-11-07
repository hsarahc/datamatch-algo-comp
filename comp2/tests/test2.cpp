#include "../scoring.hpp"

// check_compatibility returns false for users attending different colleges

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

    u1.college = "Harvard";
    u2.college = "MIT";

    // Different colleges should be incompatible
    return check_compatibility(u1, u2);
}
