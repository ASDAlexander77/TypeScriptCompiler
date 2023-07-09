#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef WIN32
#include <windows.h>
#elif _POSIX_C_SOURCE >= 199309L
#include <time.h> // for nanosleep
#include <errno.h>
#include <string.h>
#else
#include <unistd.h> // for usleep
#include <errno.h>
#include <string.h>
#endif

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if WIN32
#define POPEN _popen
#define PCLOSE _pclose
#define POPEN_MODE "rt"
#else
#define POPEN popen
#define PCLOSE pclose
#define POPEN_MODE "r"
#endif

#ifdef WIN32
#ifndef NDEBUG
#define _D_ "d"
#else
#define _D_ ""
#endif
#else
#define _D_ ""
#endif

#define ASSERT_THROW(condition)                                                                                                            \
    {                                                                                                                                      \
        if (!(condition))                                                                                                                  \
        {                                                                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") +           \
                                     std::string(__func__));                                                                               \
        }                                                                                                                                  \
    }

#define ASSERT_THROW_MSG(condition, msg)                                                                                                   \
    {                                                                                                                                      \
        if (!(condition))                                                                                                                  \
        {                                                                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") +           \
                                     std::string(__func__) + std::string(" ") + std::string(msg));                                         \
        }                                                                                                                                  \
    }

#define ASSERT_EQUAL(x, y)                                                                                                                 \
    {                                                                                                                                      \
        if ((x) != (y))                                                                                                                    \
        {                                                                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") +           \
                                     std::string(__func__) + std::string(": ") + std::to_string((x)) + std::string(" != ") +               \
                                     std::to_string((y)));                                                                                 \
        }                                                                                                                                  \
    }

#define ASSERT_EQUAL_MSG(x, y, msg)                                                                                                        \
    {                                                                                                                                      \
        if ((x) != (y))                                                                                                                    \
        {                                                                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") +           \
                                     std::string(__func__) + std::string(": ") + std::to_string((x)) + std::string(" != ") +               \
                                     std::to_string((y)) + std::string(" ") + std::string(msg));                                           \
        }                                                                                                                                  \
    }

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

void sleep_ms(int milliseconds)
{ // cross-platform sleep function
#ifdef WIN32
    Sleep(milliseconds);
#elif _POSIX_C_SOURCE >= 199309L
    struct timespec ts;
    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000;
    nanosleep(&ts, NULL);
#else
    if (milliseconds >= 1000)
        sleep(milliseconds / 1000);
    usleep((milliseconds % 1000) * 1000);
#endif
}

std::string exec(std::string cmd)
{
    auto retry = 3;

    do
    {
        std::array<char, 128> buffer;
        std::string result;

        FILE *pipe = POPEN(cmd.c_str(), POPEN_MODE);
        if (!pipe)
        {
            std::cerr << "Can't run cmd: " << cmd.c_str() << std::endl;
            std::cerr << "errno: " << strerror(errno) << std::endl;
            throw std::runtime_error("popen() failed!");
        }

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        {
            result += buffer.data();
        }

        if (feof(pipe))
        {
            auto code = PCLOSE(pipe);
            if (code)
            {
                if (retry <= 1)
                {
                    std::cerr << "Error: return code is not 0, code: " << code << " cmd: " << cmd << " output: " << result << std::endl;
                }
                else
                {
                    std::cerr << "retrying..." << std::endl;
                    sleep_ms(1000);
                    continue;
                }
            }
        }
        else
        {
            std::cerr << "Error: Failed to read the pipe to the end" << std::endl;
        }

        return result;
    } while (--retry > 0);

    return "";
}

inline bool exists(std::string name)
{
    return fs::exists(name);
}

int runFolder(const char *folder)
{
    std::string path = folder;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (!hasEnding(entry.path().extension().string(), ".ts"))
        {
            std::cout << "skipping: " << entry.path() << std::endl;
            continue;
        }

        std::cout << "Testing: " << entry.path() << std::endl;
    }

    return 0;
}

std::string readOutput(std::string fileName)
{
    std::stringstream output;

    std::ifstream fileInputStream;
    fileInputStream.open(fileName, std::fstream::in);

    std::string line;
    while (std::getline(fileInputStream, line))
    {
        output << line << std::endl;
    }

    fileInputStream.close();    

    return output.str();
}
