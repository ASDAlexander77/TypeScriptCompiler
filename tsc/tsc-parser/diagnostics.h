enum class DiagnosticCategory : int
{
    Warning,
    Error,
    Suggestion,
    Message
};

struct DiagnosticMessage
{
    DiagnosticMessage() = default;

    int code;
    DiagnosticCategory type;
    std::string label;
    std::string msg;
};

namespace Diagnostics
{
    DiagnosticMessage Merge_conflict_marker_encountered = {1185, DiagnosticCategory::Error, "Merge_conflict_marker_encountered_1185", "Merge conflict marker encountered."};
}