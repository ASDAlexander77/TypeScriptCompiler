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
    auto processCommentPragmas(SourceFile context, string sourceText) -> void;
    auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void;
    auto isExternalModule(SourceFile file) -> boolean;
    auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean;
    auto fixupParentReferences(Node rootNode) -> void;
}

#include "incremental_parser.h"

#endif // PARSER_H