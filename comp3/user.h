#ifndef USER_H
#define USER_H

#include <set>
#include <string>
#include <vector>
#include <optional>

enum Gender {
    MALE,
    FEMALE,
    NONBINARY,
    UNSPECIFIED,
};

struct User {
    std::vector<int> answers;
    std::string email;
    std::string id;
    std::string college;
    Gender gender;
    std::set<Gender> loveprefs;
    std::set<Gender> friendprefs;
    std::set<std::string> blocklist;
    int year;
    int age;
    int seriousness = 2;
    std::string crush = "";
    std::string house = "";
    std::string bio = "";
    std::string mbti = "";
    std::string costar = "";
    int ricepurity = -1;
    bool has_prompts = false;
    bool no_house_matches = false;
    std::vector<std::string> blocked_houses = {};
    std::optional<int> min_compatible_age = std::nullopt;
    std::optional<int> max_compatible_age = std::nullopt;
};

// TODO: Change to class method
std::set<Gender> get_gender_prefs(const std::string prefs_str);

#endif
