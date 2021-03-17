#include "scanner.h"

int main()
{
    ts::Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, S("function main() {}"));

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        std::wcout << scanner.syntaxKindString(token) << "(" << (int)token << S(") @") << scanner.tokenPos() << S(" '") << scanner.tokenToString(token) << "':" << scanner.tokenValue() << std::endl;
    }

    return 0;
}