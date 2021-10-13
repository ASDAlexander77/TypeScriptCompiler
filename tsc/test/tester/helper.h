#include <string>
#include <stdexcept>
#include <iostream>

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