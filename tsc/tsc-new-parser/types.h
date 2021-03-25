#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include "undefined.h"
struct DiagnosticMessageStore
{
    DiagnosticMessageStore() = default;
    DiagnosticMessageStore(int code, DiagnosticCategory category, string label, string message) : code(code), category(category), label(label), message(message) {};
    DiagnosticMessageStore(undefined_t) : category{DiagnosticCategory::Undefined} {}

    int code;
    DiagnosticCategory category;
    string label;
    string message;

    bool operator !()
    {
        return category == DiagnosticCategory::Undefined;
    }
};

#endif // ENUMS_H