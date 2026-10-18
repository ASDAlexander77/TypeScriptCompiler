#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptCompiler/Defines.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

// Borrow facts across modules (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md):
// which parameters of a library's functions their bodies only borrow, read from the library itself so
// stack promotion may put an object passed to one on the stack. Read only from a file this compile
// links or loads, because the facts must describe the code that runs: the static libraries of an
// --emit=exe link. Never from a DLL a program links, nor a -shared import - those load whatever copy
// sits beside the program at run time. Missing or unreadable facts are not an error: no facts, every
// call an escape.

extern cl::opt<enum Action> emitAction;
extern cl::list<std::string> libs;

llvm::Triple getLinkTargetTriple();
std::string getTargetStaticLibName(llvm::StringRef);
std::string getDefaultLibBinary(bool, const CompileOptions &, const llvm::Triple &);

// From TypeScript/ObjDumper.h, whose llvm/BinaryFormat/COFF.h collides with <windows.h> macros
namespace Dump
{
    std::vector<std::pair<std::string, std::string>> readCStringSymbols(llvm::StringRef, llvm::StringRef);
}

// The file a `--lib` value names, when this compile can tell: the value itself if it is a file, else
// the target's static library of that name (`<name>.lib`, `lib<name>.a`) in the current directory.
// Empty otherwise - the linker searches for it, and its facts are not read. exe.cpp links a resolved
// library by this path, so the file read is the file linked.
std::string resolveLinkLibrary(llvm::StringRef value)
{
    for (auto candidate : {value.str(), getTargetStaticLibName(value)})
    {
        if (llvm::sys::fs::is_regular_file(candidate))
        {
            llvm::SmallString<256> absolute(candidate);
            llvm::sys::fs::make_absolute(absolute);
            return absolute.str().str();
        }
    }

    return "";
}

static void appendBorrowFacts(CompileOptions &compileOptions, llvm::StringRef file)
{
    for (auto &[symbol, text] : Dump::readCStringSymbols(file, SHARED_LIB_BORROW_FACTS))
    {
        compileOptions.importedBorrowFacts += text;
        if (!text.empty() && text.back() != '\n')
        {
            compileOptions.importedBorrowFacts += '\n';
        }
    }
}

// --emit=exe: the facts of the default library's static library and of every --lib this compile can
// find. Called before the module is compiled: stack promotion reads them.
void readStaticBorrowFacts(CompileOptions &compileOptions)
{
    if (emitAction != Action::BuildExe)
    {
        return;
    }

    if (!compileOptions.noDefaultLib)
    {
        appendBorrowFacts(compileOptions, getDefaultLibBinary(/*shared=*/false, compileOptions, getLinkTargetTriple()));
    }

    for (auto &lib : libs)
    {
        auto file = resolveLinkLibrary(lib);
        if (!file.empty())
        {
            appendBorrowFacts(compileOptions, file);
        }
    }
}
