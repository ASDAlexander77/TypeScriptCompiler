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

std::string getDefaultExt(enum Action);
std::string GetTemporaryPath(llvm::StringRef, llvm::StringRef);
int dumpObjOrAssembly(int, char **, enum Action, std::string, mlir::ModuleOp, CompileOptions&);

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> getFileDeclarationContentForObjFile(llvm::StringRef objFileName) {
    llvm::SmallString<128> path(objFileName);
    llvm::sys::path::replace_extension(path, ".d.ts");

    // Handle '.d.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(path);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::warning(llvm::errs(), "tsc") << "Missing declaration file '.d.ts' for obj file: " << objFileName << " error: " << ec.message() << "\n";
    }

    return fileOrErr;
}

int declarationInline(int argc, char **argv, mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, llvm::StringRef objFileName, CompileOptions &compileOptions)
{
    // Handle '.d.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(objFileName);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SmallString<256> contentStr;
    llvm::raw_svector_ostream OS(contentStr);


    // build body of program
    OS << "@dllexport const __decls = \"";

    auto content = getFileDeclarationContentForObjFile(objFileName);
    if (!content.getError())
    {
        OS.write_escaped(content.get().get()->getBuffer());
    }

    for (auto aditionalObjFileName : objs)
    {
        auto content = getFileDeclarationContentForObjFile(aditionalObjFileName);
        if (content.getError())
        {
            // maybe we do not have declarations for it
            // TODO: print warning about it
            continue;
        }

        OS.write_escaped(content.get().get()->getBuffer());
    }

    OS << "\";";

    // ss is content
    // not we need to build obj file with one global field __decls

    auto contentBuffer = llvm::MemoryBuffer::getMemBuffer(contentStr.str(), "decl_file", false);

    auto smLoc = llvm::SMLoc();
    sourceMgr.AddNewSourceBuffer(std::move(contentBuffer), smLoc);

    const auto declFileName = "__decls.d.ts";

    // get module
    auto module = mlirGenFromSource(context, smLoc, declFileName, sourceMgr, compileOptions);
    if (!module)
    {
        return -1;
    }

    // get temp file for obj
    auto ext = getDefaultExt(Action::DumpObj);
    auto tempOutputFile = GetTemporaryPath(declFileName, ext);
    auto result = dumpObjOrAssembly(argc, argv, Action::DumpObj, tempOutputFile, *module, compileOptions);
    if (result)
    {
        return result;
    }

    return 0;
}
