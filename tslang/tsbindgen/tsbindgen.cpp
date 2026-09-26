// tsbindgen: C header -> tslang bindings. See docs/superpowers/specs/2026-09-25-tsbindgen-linkname-design.md.

#include "TsClangImporter/BindingPrinter.h"
#include "TsClangImporter/HeaderParser.h"

#include "clang/Basic/Version.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <optional>

namespace cl = llvm::cl;

namespace
{

cl::OptionCategory category("tsbindgen options");

cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.h|.c|.cpp>"), cl::Required, cl::cat(category));
cl::list<std::string> includeDirs("I", cl::Prefix, cl::desc("Add an include directory"), cl::value_desc("dir"),
                                  cl::cat(category));
cl::list<std::string> defines("D", cl::Prefix, cl::desc("Define a macro"), cl::value_desc("name[=value]"),
                              cl::cat(category));
cl::opt<std::string> target("target", cl::desc("Target triple (default: the host)"), cl::value_desc("triple"),
                            cl::cat(category));
cl::list<std::string> filters("filter", cl::desc("Emit the declarations whose C name matches this glob"),
                              cl::value_desc("glob"), cl::cat(category));
cl::opt<std::string> namespaceName("namespace", cl::desc("Wrap the output in `namespace <N>`"),
                                   cl::value_desc("N"), cl::cat(category));
cl::opt<std::string> stripPrefix("strip-prefix", cl::desc("Remove <P> from the TS names (needs --namespace)"),
                                 cl::value_desc("P"), cl::cat(category));
cl::opt<std::string> resourceDir("resource-dir", cl::desc("clang's resource directory"), cl::value_desc("dir"),
                                 cl::cat(category));
cl::opt<std::string> outputFile("o", cl::desc("Output file (default: stdout)"), cl::value_desc("out.ts"),
                                cl::init("-"), cl::cat(category));

enum ExitCode
{
    Written = 0,
    ParseOrIOError = 1,
    BadArguments = 2
};

std::string version()
{
#ifdef TSBINDGEN_VERSION
    return TSBINDGEN_VERSION;
#else
    return "dev";
#endif
}

bool hasStddef(llvm::StringRef dir)
{
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, "include", "stddef.h");
    return llvm::sys::fs::exists(path);
}

// clang's own stddef.h/stdint.h live here; without them those headers do not resolve.
std::optional<std::string> findResourceDir(llvm::StringRef exePath, std::vector<std::string> &tried)
{
    auto versioned = [](llvm::StringRef base) {
        llvm::SmallString<256> path(base);
        llvm::sys::path::append(path, "lib", "clang", std::to_string(CLANG_VERSION_MAJOR));
        return std::string(path);
    };

    std::vector<std::string> candidates;
    if (!resourceDir.empty())
    {
        candidates.push_back(resourceDir);
    }
    else
    {
        auto exeDir = llvm::sys::path::parent_path(exePath);
        candidates.push_back(versioned(exeDir));                              // the release package
        candidates.push_back(versioned(llvm::sys::path::parent_path(exeDir))); // an LLVM-style install
#ifdef TSBINDGEN_CONFIGURED_RESOURCE_DIR
        candidates.push_back(TSBINDGEN_CONFIGURED_RESOURCE_DIR); // the LLVM this was built with
#endif
    }

    for (auto &candidate : candidates)
    {
        tried.push_back(candidate);
        if (hasStddef(candidate))
        {
            return candidate;
        }
    }

    return std::nullopt;
}

bool isCxx(llvm::StringRef path)
{
    auto extension = llvm::sys::path::extension(path).lower();
    return extension == ".cpp" || extension == ".cc" || extension == ".cxx" || extension == ".hpp" ||
           extension == ".hh" || extension == ".hxx";
}

} // namespace

int main(int argc, char **argv)
{
    // everything after `--` goes to clang untouched
    std::vector<const char *> ownArgs;
    std::vector<std::string> extraArgs;
    auto afterDashes = false;
    for (int index = 0; index < argc; ++index)
    {
        if (afterDashes)
        {
            extraArgs.push_back(argv[index]);
        }
        else if (index > 0 && llvm::StringRef(argv[index]) == "--")
        {
            afterDashes = true;
        }
        else
        {
            ownArgs.push_back(argv[index]);
        }
    }

    cl::HideUnrelatedOptions(category);
    cl::SetVersionPrinter([](llvm::raw_ostream &os) {
        os << "tsbindgen " << version() << " (clang " << CLANG_VERSION_STRING << ")\n";
    });

    if (!cl::ParseCommandLineOptions(static_cast<int>(ownArgs.size()), ownArgs.data(),
                                     "C header -> tslang bindings\n", &llvm::errs()))
    {
        return BadArguments;
    }

    if (!stripPrefix.empty() && namespaceName.empty())
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << "--strip-prefix needs --namespace\n";
        return BadArguments;
    }

    std::vector<std::string> tried;
    auto exePath = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
    auto foundResourceDir = findResourceDir(exePath, tried);
    if (!foundResourceDir)
    {
        auto &error = llvm::WithColor::error(llvm::errs(), "tsbindgen");
        error << "clang's resource directory not found; tried:";
        for (auto &path : tried)
        {
            error << "\n  " << path;
        }

        error << "\n(pass --resource-dir <dir>, the directory with include/stddef.h)\n";
        // a wrong --resource-dir is the caller's argument; a failed lookup is a broken install
        return resourceDir.empty() ? ParseOrIOError : BadArguments;
    }

    auto buffer = llvm::MemoryBuffer::getFile(inputFile);
    if (!buffer)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen")
            << "cannot read '" << inputFile << "': " << buffer.getError().message() << "\n";
        return ParseOrIOError;
    }

    llvm::SmallString<256> absoluteInput(inputFile);
    llvm::sys::fs::make_absolute(absoluteInput);

    auto triple = target.empty() ? llvm::sys::getDefaultTargetTriple() : target.getValue();

    tsbindgen::ParseInput input;
    input.code = (*buffer)->getBuffer().str();
    input.fileName = std::string(absoluteInput);
    input.args = {"-x", isCxx(inputFile) ? "c++" : "c", "--target=" + triple, "-resource-dir", *foundResourceDir};
    for (auto &dir : includeDirs)
    {
        input.args.push_back("-I" + dir);
    }

    for (auto &define : defines)
    {
        input.args.push_back("-D" + define);
    }

    input.args.insert(input.args.end(), extraArgs.begin(), extraArgs.end());

    auto model = tsbindgen::parseHeader(input);
    if (!model)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << llvm::toString(model.takeError()) << "\n";
        return ParseOrIOError;
    }

    std::string commandLine = "tsbindgen";
    for (int index = 1; index < argc; ++index)
    {
        commandLine += " ";
        commandLine += argv[index];
    }

    tsbindgen::PrintOptions options;
    options.filters = filters;
    options.namespaceName = namespaceName;
    options.stripPrefix = stripPrefix;
    options.headerLines = {"Generated by tsbindgen " + version() + " (clang " CLANG_VERSION_STRING "); do not edit.",
                           commandLine};
    // `import` would not do: it loads a same-named .dll/.so as a tslang library if there is one,
    // and includes a source file as declarations only, so its constants would have no value
    if (outputFile != "-")
    {
        options.headerLines.push_back("Include with: /// <reference path=\"" +
                                      llvm::sys::path::filename(outputFile).str() + "\" />");
    }

    auto printed = tsbindgen::printBindings(*model, options);
    if (!printed)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << llvm::toString(printed.takeError()) << "\n";
        return BadArguments;
    }

    for (auto &warning : printed->warnings)
    {
        llvm::WithColor::warning(llvm::errs(), "tsbindgen") << warning << "\n";
    }

    // A .d.ts is included as declarations only: a `const` in it has no value and fails to link.
    if (llvm::StringRef(outputFile.getValue()).ends_with(".d.ts") &&
        llvm::StringRef(printed->text).contains("const "))
    {
        llvm::WithColor::warning(llvm::errs(), "tsbindgen")
            << "'" << outputFile << "' declares constants, which a .d.ts file cannot define; name it .ts\n";
    }

    std::error_code error;
    llvm::raw_fd_ostream out(outputFile, error, llvm::sys::fs::OF_Text);
    if (error)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen")
            << "cannot write '" << outputFile << "': " << error.message() << "\n";
        return ParseOrIOError;
    }

    out << printed->text;
    out.close();
    if (out.has_error())
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << "cannot write '" << outputFile << "'\n";
        out.clear_error();
        return ParseOrIOError;
    }

    return Written;
}
