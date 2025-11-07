#include "../scoring.hpp"

// check_compatibility returns false for the same user

int main() {
    User u1;

    u1.id = 1;

    return check_compatibility(u1, u1); 
}
