#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptCompiler/Defines.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

// Borrow facts across modules (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md):
// which parameters of a library's functions their bodies only borrow, read from the library itself so
// stack promotion may put an object passed to one on the stack. Read only from a file this compile
// links or loads, because the facts must describe the code that runs: the static libraries of an
// --emit=exe link, and the default library's DLL the JIT loads. Never from a DLL a program links,
// nor a -shared import - those load whatever copy sits beside the program at run time. Missing or
// unreadable facts are not an error: no facts, every call an escape.

extern cl::opt<enum Action> emitAction;
extern cl::list<std::string> libs;
extern cl::list<std::string> objs;
extern cl::list<std::string> ownFacts;

llvm::Triple getLinkTargetTriple();
std::string getTargetStaticLibName(llvm::StringRef);
std::string getDefaultLibBinary(bool, const CompileOptions &, const llvm::Triple &);

// From TypeScript/ObjDumper.h, whose llvm/BinaryFormat/COFF.h collides with <windows.h> macros
namespace Dump
{
    std::vector<std::pair<std::string, std::string>> readCStringSymbols(llvm::StringRef, llvm::StringRef);
    std::vector<std::string> definedSymbols(llvm::StringRef);
}

// A static library or an object file: linked into the program, so what it publishes describes the
// code that runs. Not a DLL or shared object, whose copy at run time may be another.
static bool isStaticLinkInput(llvm::StringRef file)
{
    llvm::file_magic magic;
    if (llvm::identify_magic(file, magic))
    {
        return false;
    }

    return magic == llvm::file_magic::archive || magic == llvm::file_magic::coff_object ||
           magic == llvm::file_magic::elf_relocatable;
}

// The static library a `--lib` value names, when this compile can tell: the value itself if it is
// one, else the target's static library of that name (`<name>.lib`, `lib<name>.a`) in the current
// directory. Empty otherwise - the linker searches for it, or it is linked dynamically, and then no
// borrow facts are trusted at all (readBorrowFacts). exe.cpp links a resolved library by this path,
// so the file read is the file linked.
std::string resolveLinkLibrary(llvm::StringRef value)
{
    for (auto candidate : {value.str(), getTargetStaticLibName(value)})
    {
        if (llvm::sys::fs::is_regular_file(candidate) && isStaticLinkInput(candidate))
        {
            llvm::SmallString<256> absolute(candidate);
            llvm::sys::fs::make_absolute(absolute);
            return absolute.str().str();
        }
    }

    return "";
}

// The facts of `files`, each line kept only when no other of `files` or `others` defines its symbol:
// the linker may take that other definition, which the fact does not describe.
static void appendBorrowFacts(CompileOptions &compileOptions, llvm::ArrayRef<std::string> files,
                              llvm::ArrayRef<std::string> others = {})
{
    llvm::SmallVector<llvm::StringSet<>> defined;
    for (auto &file : files)
    {
        defined.emplace_back();
        for (auto &symbol : Dump::definedSymbols(file))
        {
            defined.back().insert(symbol);
        }
    }

    llvm::StringSet<> definedByOthers;
    for (auto &file : others)
    {
        for (auto &symbol : Dump::definedSymbols(file))
        {
            definedByOthers.insert(symbol);
        }
    }

    for (auto [index, file] : llvm::enumerate(files))
    {
        if (llvm::sys::fs::exists(file))
        {
            compileOptions.importedBorrowFactsFiles.push_back(file);
        }

        for (auto &[symbol, text] : Dump::readCStringSymbols(file, SHARED_LIB_BORROW_FACTS))
        {
            llvm::SmallVector<llvm::StringRef> lines;
            llvm::StringRef(text).split(lines, '\n', -1, /*KeepEmpty=*/false);
            for (auto line : lines)
            {
                auto name = line.split(' ').first;
                auto elsewhere = definedByOthers.contains(name);
                for (auto other : llvm::seq<size_t>(0, files.size()))
                {
                    elsewhere |= other != index && defined[other].contains(name);
                }

                if (!elsewhere)
                {
                    compileOptions.importedBorrowFacts += line.str() + "\n";
                }
            }
        }
    }
}

// Called before the module is compiled: stack promotion reads them.
// --emit=exe: the facts of the default library's static library and of every --lib this compile can
// find. --emit=jit: those of the default library's DLL the JIT loads, read from its file.
void readBorrowFacts(CompileOptions &compileOptions)
{
    if (emitAction == Action::RunJIT)
    {
        if (!compileOptions.noDefaultLib)
        {
            // the JIT is host-only, and loads this file (jit.cpp, prepareJitProcess)
            appendBorrowFacts(compileOptions,
                              {getDefaultLibBinary(/*shared=*/true, compileOptions,
                                                   llvm::Triple(llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple())))});
        }

        return;
    }

    if (emitAction != Action::BuildExe)
    {
        return;
    }

    std::vector<std::string> files;
    if (!compileOptions.noDefaultLib)
    {
        files.push_back(getDefaultLibBinary(/*shared=*/false, compileOptions, getLinkTargetTriple()));
    }

    for (auto &lib : libs)
    {
        auto file = resolveLinkLibrary(lib);
        if (file.empty())
        {
            // one the linker searches for itself may define anything, under any name
            return;
        }

        files.push_back(file);
    }

    appendBorrowFacts(compileOptions, files, std::vector<std::string>(objs.begin(), objs.end()));
}

// Called before the module is compiled: the ownership signature pass reads them (MLIRGen puts
// them on the module). Only under own, and only from the files --own-facts names: the objects of
// the modules this one imports, which the build links with it - so, like a library's borrow facts,
// they describe the code that runs. A file that is missing or holds none gives no facts.
void readOwnFacts(CompileOptions &compileOptions)
{
    if (compileOptions.memoryModel != MemoryModelOwn)
    {
        return;
    }

    // and the default library's: its own build is the one every own program links (statically: its
    // DLL, which a program may load another copy of, is never read), so `Math.abs` is a call of a
    // function whose facts are known. Under the JIT, the DLL the JIT loads is the code that runs:
    // its facts are read from its file, as its borrow facts are (readBorrowFacts). Without them
    // every call into the library may destroy anything.
    std::vector<std::string> files(ownFacts.begin(), ownFacts.end());
    std::string jitDefaultLib;
    if (!compileOptions.noDefaultLib)
    {
        if (emitAction == Action::RunJIT)
        {
            // the JIT is host-only, and loads this file (jit.cpp, prepareJitProcess)
            jitDefaultLib = getDefaultLibBinary(/*shared=*/true, compileOptions,
                                                llvm::Triple(llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple())));
            files.push_back(jitDefaultLib);
        }
        else
        {
            files.push_back(getDefaultLibBinary(/*shared=*/false, compileOptions, getLinkTargetTriple()));
        }
    }

    for (auto &file : files)
    {
        if (!llvm::sys::fs::is_regular_file(file) || (!isStaticLinkInput(file) && file != jitDefaultLib))
        {
            continue;
        }

        // a JIT cache entry depends on what they said
        compileOptions.importedOwnFactsFiles.push_back(file);

        for (auto &[symbol, text] : Dump::readCStringSymbols(file, SHARED_LIB_OWN_OBJECT_FACTS))
        {
            compileOptions.importedOwnFacts += text;
            if (!text.empty() && text.back() != '\n')
            {
                compileOptions.importedOwnFacts += "\n";
            }
        }
    }
}
