#ifndef TSLANG_LINK_H_
#define TSLANG_LINK_H_

#include "TypeScript/DataStructs.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

// What linkBinary links: the objects, in the order given, into `output`, with the libraries the
// compile options call for - the link `--emit=exe` and `--emit=dll` make, without the command line
// they read their objects and output from.
struct LinkRequest
{
    llvm::SmallVector<std::string> objects;
    std::string output;
    // a DLL or shared object rather than a program
    bool shared = false;
    // a shared library that links the static default library, as a program does, rather than
    // importing the default library's DLL
    bool staticDefaultLib = false;
    // driver arguments added after tslang's own, e.g. -Wl,/EXPORT:name
    llvm::SmallVector<std::string> extraArgs;
    // the linker's output is shown only if the link fails
    bool quiet = false;
};

// Links with the clang driver, as buildExe does; callable more than once in a process. Non-zero if
// the link fails. `argv0` is tslang's own path, which the driver resolves its tools against.
int linkBinary(const char *argv0, const LinkRequest &request, CompileOptions &compileOptions);

#endif // TSLANG_LINK_H_
