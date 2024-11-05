#include "TypeScript/MLIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileUtilities.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#include <regex>

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::list<std::string> objs;

std::string getFileDeclarationContentForObjFile(std::string objFileName) {
    llvm::SmallString<128> path(objFileName);
    llvm::sys::path::replace_extension(path, ".d.ts");

    // Handle '.d.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(path);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open obj file: " << ec.message() << "\n";
        return {};
    }

    auto &f = *fileOrErr.get();

    auto *file1Start = f.getBufferStart();
    //auto *file1End = f.getBufferEnd();
    auto size = f.getBufferSize();

    return std::string(file1Start, size);
}

int declarationInline(int argc, char **argv, std::string objFileName, CompileOptions &compileOptions)
{
    // Handle '.d.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(objFileName);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    std::stringstream ss;

    // build body of program
    ss << "@dllexport const __decls = '";

    auto content = getFileDeclarationContentForObjFile(objFileName);
    if (!content.empty())
    {
        // maybe we do not have declarations for it
        // TODO: print warning about it
        ss << std::regex_replace(content, std::regex("'"), "\'");
    }

    for (auto aditionalObjFileName : objs)
    {
        auto content = getFileDeclarationContentForObjFile(objFileName);
        if (content.empty())
        {
            // maybe we do not have declarations for it
            // TODO: print warning about it
            continue;
        }

        ss << std::regex_replace(content, std::regex("'"), "\'");
    }

    ss << "';";

    // ss is content
    // not we need to build obj file with one global field __decls

    return 0;
}
