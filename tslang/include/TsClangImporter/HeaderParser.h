#ifndef TSCLANGIMPORTER_HEADERPARSER_H
#define TSCLANGIMPORTER_HEADERPARSER_H

#include "TsClangImporter/BindingModel.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace tsbindgen
{

struct ParseInput
{
    std::string code;             // the input file's contents
    std::string fileName;         // its path; `#include "x.h"` resolves next to it
    std::vector<std::string> args; // clang arguments: -x, --target=, -resource-dir, -I, -D, ...
    // extra files visible to clang only (tests use them in place of real headers)
    std::vector<std::pair<std::string, std::string>> virtualFiles;
};

// Runs clang over the input and maps what it declares. clang's diagnostics go to stderr as clang
// prints them; an error means clang reported an error.
llvm::Expected<HeaderModel> parseHeader(const ParseInput &input);

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_HEADERPARSER_H
