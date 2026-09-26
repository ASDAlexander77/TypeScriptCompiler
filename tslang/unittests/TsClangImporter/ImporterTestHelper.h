#ifndef TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H
#define TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H

#include "TsClangImporter/BindingPrinter.h"
#include "TsClangImporter/HeaderParser.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <vector>

namespace tsbindgen_test
{

inline const char *const windowsTarget = "x86_64-pc-windows-msvc";
inline const char *const linuxTarget = "x86_64-linux-gnu";

struct Generated
{
    std::string text;
    std::vector<std::string> warnings;
};

// Parses `header` as the file input.h and prints its bindings. Hermetic: -ffreestanding and
// -nostdlibinc leave only clang's own headers (stdint.h, stddef.h, ...), from the resource directory
// of the LLVM the tests were built with, so no system SDK is needed and one host checks any target.
// `files` are extra headers input.h can include; a path under "sys/" is a system header.
inline Generated generate(const std::string &header, tsbindgen::PrintOptions options = {},
                          const std::string &target = windowsTarget,
                          std::vector<std::pair<std::string, std::string>> files = {},
                          const std::string &language = "c")
{
    tsbindgen::ParseInput input;
    input.code = header;
    input.fileName = "input.h";
    input.args = {"-x", language, "--target=" + target, "-ffreestanding", "-nostdlibinc", "-isystem", "sys",
                  "-resource-dir", TSBINDGEN_TEST_RESOURCE_DIR};
    input.virtualFiles = std::move(files);

    auto model = tsbindgen::parseHeader(input);
    if (!model)
    {
        ADD_FAILURE() << llvm::toString(model.takeError());
        return {};
    }

    auto printed = tsbindgen::printBindings(*model, options);
    if (!printed)
    {
        ADD_FAILURE() << llvm::toString(printed.takeError());
        return {};
    }

    return {printed->text, printed->warnings};
}

inline std::string text(const std::string &header, const std::string &target = windowsTarget)
{
    return generate(header, {}, target).text;
}

} // namespace tsbindgen_test

#endif // TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H
