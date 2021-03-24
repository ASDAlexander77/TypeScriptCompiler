#ifndef PARSER_JDOC_H
#define PARSER_JDOC_H

#include "config.h"
#include "parser.h"

struct EntityNameWIthBracketed 
{ 
    EntityNameWIthBracketed() = default;
    EntityName name;
    boolean isBracketed;
};

#endif // PARSER_JDOC_H