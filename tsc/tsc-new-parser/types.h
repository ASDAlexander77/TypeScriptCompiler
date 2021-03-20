#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include "undefined.h"

struct DiagnosticMessage
{
    DiagnosticMessage() = default;

    int code;
    DiagnosticCategory category;
    string label;
    string message;
};

struct TextRange {
    TextRange() = default;
    number pos;
    number end;
};

struct CommentDirective {
    CommentDirective() = default;
    TextRange range;
    CommentDirectiveType type;
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
    DiagnosticRelatedInformation() = default;
    
    DiagnosticRelatedInformation(undefined_t) : category(DiagnosticCategory::Undefined) {}

    DiagnosticCategory category;
    string fileName;
    number code;
    number start;
    number length;
    string messageText;

    inline boolean operator !()
    {
        return category == DiagnosticCategory::Undefined;
    }
};

struct Diagnostic : DiagnosticRelatedInformation {
    std::vector<string> reportsUnnecessary;
    std::vector<DiagnosticRelatedInformation> relatedInformation;
};

struct DiagnosticWithDetachedLocation : Diagnostic {

};


#endif // ENUMS_H