#ifndef TYPES_H
#define TYPES_H

#include "config.h"

struct DiagnosticMessage
{
    DiagnosticMessage() = default;

    int code;
    DiagnosticCategory category;
    string label;
    string message;
};

struct TextSpan {
    number start;
    number length;
};

struct FileReference : TextSpan {
    string fileName;
};

struct AmdDependency {
    string path;
    string name;
};

struct TextChangeRange {
    TextSpan span;
    number newLength;
};

struct DiagnosticRelatedInformation {
    DiagnosticCategory category;
    string fileName;
    number code;
    number start;
    number length;
    string messageText;
};

struct Diagnostic : DiagnosticRelatedInformation {
    std::vector<string> reportsUnnecessary;
    std::vector<DiagnosticRelatedInformation> relatedInformation;
};

struct DiagnosticWithDetachedLocation : Diagnostic {
};


#endif // ENUMS_H