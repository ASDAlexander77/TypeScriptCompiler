#ifndef UNDEFINED_H
#define UNDEFINED_H

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

    bool _hasValue;
    T _value;

    operator bool()
    {
        if (!_hasValue)
        {
            return false;
        }

        return !!_value;
    }

    operator T&()
    {
        return _value;
    }

    bool hasValue()
    {
        return _hasValue;
    }
};

#endif // UNDEFINED_H