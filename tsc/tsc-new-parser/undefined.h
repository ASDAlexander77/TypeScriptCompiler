#ifndef UNDEFINED_H
#define UNDEFINED_H

#include "config.h"

struct undefined_t
{
    constexpr operator number()
    {
        return -1;
    }
};

static undefined_t undefined;


#endif // UNDEFINED_H