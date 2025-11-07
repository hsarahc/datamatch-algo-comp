#include <vector>
#include <unordered_set>
#include <string>
#include <optional>

enum Gender {
    MALE,
    FEMALE,
    NONBINARY,
    UNSPECIFIED,
};

struct User {
    int id = -1;

    Gender gender;
    std::unordered_set<Gender> loveprefs;
    std::unordered_set<Gender> friendprefs;

    std::string college;
    std::string house;
    bool no_house_matches = false;
    std::vector<std::string> blocked_houses;

    int year = 0; // academic year bucket
    int age = 0;

    std::optional<int> min_compatible_age;
    std::optional<int> max_compatible_age;

    int crush = -1; // partner id, if any

    std::string bio;
    bool has_prompts = false;

    float ricepurity = -1.0f; // -1 means unknown

    int seriousness = 0; // 0..3 scale

    std::vector<int> answers;
};