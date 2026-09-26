#ifndef TSCLANGIMPORTER_BINDINGPRINTER_H
#define TSCLANGIMPORTER_BINDINGPRINTER_H

#include "TsClangImporter/BindingModel.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace tsbindgen
{

struct PrintOptions
{
    // A declaration is selected if its C name matches one of these globs, or, with none, if the
    // input file itself declares it. Types a selected declaration uses are emitted too.
    std::vector<std::string> filters;
    std::string namespaceName; // --namespace: wrap in `namespace N { export ... }`, with @linkname
    std::string stripPrefix;   // --strip-prefix: removed from the TS names (needs a namespace)
    std::vector<std::string> headerLines; // printed first, each as a `//` comment
};

struct PrintResult
{
    std::string text;
    std::vector<std::string> warnings; // one per skipped declaration, and per name kept unstripped
};

// Fails only for a filter that is not a valid glob.
llvm::Expected<PrintResult> printBindings(const HeaderModel &model, const PrintOptions &options);

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_BINDINGPRINTER_H
