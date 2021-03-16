#include "scanner.h"

int main()
{
    ts::Scanner scanner(ScriptTarget::Latest, true, LanguageVariant::Standard, S("function main() {}"));
    return 0;
}