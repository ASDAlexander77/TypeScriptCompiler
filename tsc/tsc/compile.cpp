#include "TypeScript/MLIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<bool> disableGC;

int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
    auto fileName = llvm::StringRef(inputFilename);

    // Handle '.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    CompileOptions compileOptions;
    compileOptions.disableGC = disableGC;
    module = mlirGenFromSource(context, fileName, fileOrErr.get()->getBuffer(), compileOptions);
    return !module ? 1 : 0;
}
