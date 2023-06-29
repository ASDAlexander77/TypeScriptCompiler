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

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<bool> disableGC;
extern cl::opt<bool> disableWarnings;
extern cl::opt<bool> generateDebugInfo;
extern cl::opt<bool> lldbDebugInfo;
extern cl::opt<std::string> TargetTriple;

int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
    auto fileName = llvm::StringRef(inputFilename);

    // Handle '.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    auto moduleTargetTriple = TargetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : llvm::Triple::normalize(TargetTriple);

    CompileOptions compileOptions;
    compileOptions.disableGC = disableGC;
    compileOptions.disableWarnings = disableWarnings;
    compileOptions.generateDebugInfo = generateDebugInfo;
    compileOptions.lldbDebugInfo = lldbDebugInfo;
    compileOptions.moduleTargetTriple = moduleTargetTriple;

    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    module = mlirGenFromSource(context, fileName, sourceMgr, compileOptions);
    return !module ? 1 : 0;
}
