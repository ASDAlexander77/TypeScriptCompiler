#include "scanner.h"

int main()
{
    ts::Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, S("function main() {}"));

    auto token = SyntaxKind::Unknown;
    while (token != SyntaxKind::EndOfFileToken)
    {
        token = scanner.scan();
        std::cout << (int)token << std::endl;
    }

    return 0;
}