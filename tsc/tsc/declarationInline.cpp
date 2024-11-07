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

#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptCompiler/Defines.h"

#include <regex>

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::list<std::string> objs;
extern cl::opt<bool> verbose;

std::string getDefaultExt(enum Action);
std::string getDefaultOutputFileName(enum Action);
std::string GetTemporaryPath(llvm::StringRef, llvm::StringRef);
int runMLIRPasses(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &, CompileOptions&);
int dumpObjOrAssembly(int, char **, enum Action, std::string, mlir::ModuleOp, CompileOptions&);
int dumpLLVMIR(enum Action, std::string, mlir::ModuleOp, CompileOptions&);

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

int declarationInline(int argc, char **argv, mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, llvm::StringRef tsFileName, CompileOptions &compileOptions, std::string& tempOutputFile)
{
    // Handle '.d.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(tsFileName);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SmallString<256> contentStr;
    llvm::raw_svector_ostream OS(contentStr);

    // build body of program
    OS << "export const " SHARED_LIB_DECLARATIONS_2UNDERSCORE " = \"";
    
    auto content = getFileDeclarationContentForObjFile(tsFileName);
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

    auto contentBuffer = llvm::MemoryBuffer::getMemBuffer(contentStr.str(), SHARED_LIB_DECLARATIONS_FILENAME, false);

    LLVM_DEBUG(llvm::dbgs() << "declaration content: " << contentStr.str() << "\n";);

    auto smLoc = llvm::SMLoc::getFromPointer(contentBuffer->getBufferStart());
    sourceMgr.AddNewSourceBuffer(std::move(contentBuffer), smLoc);

    // !!! do not use .d.ts - or all variables will be declared 'external' and will not be resolved
    const auto declFileName = SHARED_LIB_DECLARATIONS_FILENAME;

    // get module
    auto module = mlirGenFromSource(context, smLoc, declFileName, sourceMgr, compileOptions);
    if (!module)
    {
        return -1;
    }

    if (int error = runMLIRPasses(context, sourceMgr, module, compileOptions))
    {
        return error;
    }    

    // get temp file for obj
    auto ext = getDefaultExt(Action::DumpObj);
    tempOutputFile = GetTemporaryPath(declFileName, ext);
    auto result = dumpObjOrAssembly(argc, argv, Action::DumpObj, tempOutputFile, *module, compileOptions);
    if (result)
    {
        return result;
    }

    if (verbose.getValue())
    {
        dumpLLVMIR(Action::DumpLLVMIR, getDefaultOutputFileName(Action::DumpLLVMIR), *module, compileOptions);
        dumpObjOrAssembly(argc, argv, Action::DumpAssembly, getDefaultOutputFileName(Action::DumpAssembly), *module, compileOptions);
    }

    return 0;
}
