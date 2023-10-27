#ifndef DEBUG_H
#define DEBUG_H

#include "parser.h"

namespace ts
{
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

        template <typename T>
        static auto assertEqual(T &l, T &r) -> void
        {
            assert(l == r);
        }

        static auto assertGreaterThanOrEqual(number l, number r) -> void
        {
            assert(l >= r);
        }

        static auto assertLessThanOrEqual(number l, number r) -> void
        {
            assert(l <= r);
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

        template <typename T>
        static auto _assertNever(T value)
        {
            assert(false);
        }    

        template <typename T>
        static auto fail(string message = string()) -> T {
            //debugger;
            string msg = S("Debug Failure.");
            if (!message.empty())
            {
                msg += message;
            }

            throw msg;
        }

        template<typename T>
        static auto assertIsDefined(T value, string message = string()) -> T {
            // eslint-disable-next-line no-null/no-null
            if (value == undefined || value == nullptr) {
                fail<void>(message);
            }

            return value;
        }

        template<typename T>
        static auto checkDefined(T value, string message = string()) -> T {
            assertIsDefined(value, message);
            return value;
        }

        template<typename T>
        static auto attachNodeArrayDebugInfo(NodeArray<T> &elements) -> void {
        }
    };
} // namespace ts

#endif // DEBUG_H