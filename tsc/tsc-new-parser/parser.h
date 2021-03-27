#ifndef PARSER_H
#define PARSER_H

#include "undefined.h"
#include "enums.h"
#include "types.h"
#include "scanner.h"
#include "debug.h"

#include <memory>

#include "parser_types.h"

template <typename T>
using NodeFuncT = std::function<T(Node)>;
template <typename T>
using NodeWithParentFuncT = std::function<T(Node, Node)>;

typedef std::function<Node(SyntaxKind, number, number)> NodeCreateFunc;

typedef std::function<void(number, number, DiagnosticMessage)> PragmaDiagnosticReporter;

namespace ts
{
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

        auto Parser::parseSourceFile(string, ScriptTarget) -> SourceFile;

        auto Parser::parseSourceFile(string, string, ScriptTarget, IncrementalParser::SyntaxCursor, boolean = false, ScriptKind = ScriptKind::Unknown) -> SourceFile;

        ~Parser();
    };

}

#include "incremental_parser.h"

#endif // PARSER_H