#ifndef DEBUG_H
#define DEBUG_H

#include "parser.h"

enum class AssertionLevel : number {
    None = 0,
    Normal = 1,
    Aggressive = 2,
    VeryAggressive = 3,
};

struct Debug
{
    static AssertionLevel currentAssertionLevel;

    static auto shouldAssert(AssertionLevel level) -> boolean
    {
        return currentAssertionLevel >= level;
    }

    static void _assert(boolean cond)
    {
        assert(cond);
    }

    static void _assert(string msg)
    {
        std::wcerr << msg.c_str();
    }

    static void _assert(boolean cond, string msg)
    {
        if (!cond)
        {
            std::wcerr << msg.c_str();
        }
        
        assert(cond);
    }

};

#endif // DEBUG_H