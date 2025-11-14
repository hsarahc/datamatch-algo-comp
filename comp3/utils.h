#ifndef UTILS_H
#define UTILS_H

#include "lib/json.hpp"
#include <unordered_map>

using json = nlohmann::json;

enum LogLevel {
    INFO,
    DEBUG,
    WARNING,
    ERROR,
};

/**
 * @brief This Logging class follows the Singleton design pattern,
 * with the intent of maintaining only a single instance of this class.
 * Operations may not be threadsafe due to its global nature, just as a warning.
 *
 * Initializing the Logger is through the init method:
 *    Logger logger = Logger::init(
 *      Logger::create_file(),
 *      "[datamatch_algo] "
 *    );
 *
 * where you can either specify the file using fopen() or use the static method
 * create_file(). Furthermore, you can also customize the header of each line
 * the logger outputs.
 *
 * Accessing the logger instance is done through the instance() method:
 *    Logger logger = Logger::instance();
 *
 * The log() method also takes in a 'level' argument, which simply appends
 * a level label to each log line. The given options consist of INFO,
 * DEBUG, WARNING, and ERROR.
 *
 * Example log output:
 *    [datamatch_algo] INFO: Beginning algorithm
 *
 */
class Logger {
public:
    FILE* out { nullptr };
    std::unordered_map<LogLevel, std::string> headers = {
        {DEBUG, "Debug: "},
        {WARNING, "Warning: "},
        {ERROR, "Error: "}
    };

    static Logger& init(FILE* out);
    static Logger& instance();
    void log(LogLevel level, const char* format, ...);
    static FILE* create_file();

private:
    static Logger logger;
};

// get_index(v, target)
//   Return the index of the first occurrence of `target` in `v`. If `target`
//   does not exist, return the size of `v`.
size_t get_index(const std::vector<std::string> v, const std::string target);

// normal_cdf(x);
//   Get the Normal CDF.
float normal_cdf(float x);

// validate_json(f, required_fields)
//   Verify that the JSON file `f` contains all of `required_fields`.
bool validate_json(const json f,
                   const std::vector<std::string>& required_fields);

// sort_on_other(main, other, size)
//   Sort `main` based on the sorted order of `other`.
void sort_on_other(int* main, int* other, int size);

std::vector<std::string> split(std::string str, const char delim);

#endif
