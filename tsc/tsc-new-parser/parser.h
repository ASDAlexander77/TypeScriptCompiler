#ifndef PARSER_H
#define PARSER_H

#include "undefined.h"
#include "enums.h"
#include "types.h"
#include "scanner.h"
#include "debug.h"

#include <memory>

#include "parser_types.h"

namespace ts
{

template <typename R = Node, typename T = Node> using FuncT = std::function<R(T)>;
template <typename R = Node, typename T = Node> using FuncWithParentT = std::function<R(T, T)>;

typedef std::function<Node(SyntaxKind, pos_type, number)> NodeCreateFunc;

typedef std::function<void(pos_type, number, DiagnosticMessage)> PragmaDiagnosticReporter;

typedef std::function<void(Node)> NodeCreateCallbackFunc;

namespace IncrementalParser
{
struct SyntaxCursor;
}

auto processCommentPragmas(SourceFile context, string sourceText) -> void;
auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void;
auto isExternalModule(SourceFile file) -> boolean;
auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean;
auto fixupParentReferences(Node rootNode) -> void;

namespace Impl
{
struct Parser;
};

class Parser
{
    Impl::Parser *impl;

  public:
    Parser();

    auto parseSourceFile(string, ScriptTarget) -> SourceFile;

    auto parseSourceFile(string, string, ScriptTarget) -> SourceFile;

    auto parseSourceFile(string, string, ScriptTarget, IncrementalParser::SyntaxCursor, boolean = false, ScriptKind = ScriptKind::Unknown)
        -> SourceFile;

    auto tokenToText(SyntaxKind kind) -> string;

    auto syntaxKindString(SyntaxKind kind) -> string;

    auto getLineAndCharacterOfPosition(SourceFileLike sourceFile, number position) -> LineAndCharacter;

    ~Parser();
};

} // namespace ts

#include "incremental_parser.h"

#endif // PARSER_H