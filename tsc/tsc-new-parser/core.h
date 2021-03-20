#ifndef CORE_H
#define CORE_H

#include "config.h"

struct undefined_t
{
};

static undefined_t undefined;

template <typename T>
struct Undefined
{
    Undefined() : _hasValue(false)
    {
    }

    Undefined(undefined_t) : _hasValue(false)
    {
    }

    Undefined(T value) : _hasValue(true), _value(value)
    {
    }

    boolean _hasValue;
    T _value;

    operator bool()
    {
        if (!_hasValue)
        {
            return false;
        }

        return !!_value;
    }

    bool hasValue()
    {
        return _hasValue;
    }
};

#endif // CORE_H