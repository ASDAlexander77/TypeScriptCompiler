#include "TypeScript/MLIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileUtilities.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;

int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, mlir::OwningOpRef<mlir::ModuleOp> &module, CompileOptions &compileOptions)
{
    auto fileName = llvm::StringRef(inputFilename);
    
    llvm::SmallString<128> absoluteFilePath("");

    if (fileName != "-") {
        llvm::SmallString<128> initialFilePath(fileName);
        llvm::sys::fs::real_path(initialFilePath, absoluteFilePath);
    } else {
        absoluteFilePath = fileName;
    }

    // Handle '.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(absoluteFilePath);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }
    
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    module = mlirGenFromSource(context, absoluteFilePath, sourceMgr, compileOptions);
    return !module ? 1 : 0;
}
