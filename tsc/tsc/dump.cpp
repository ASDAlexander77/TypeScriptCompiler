#include "TypeScript/MLIRGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::opt<bool> lldbDebugInfo;
extern cl::opt<std::string> outputFilename;

std::string getDefaultOutputFileName(enum Action);
std::unique_ptr<llvm::ToolOutputFile> getOutputStream(enum Action, std::string);
int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int, CompileOptions&);

int dumpAST()
{
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return 0;
    }

    llvm::outs() << dumpFromSource(inputFilename, fileOrErr.get()->getBuffer());

    return 0;
}

int dumpLLVMIR(enum Action emitAction, std::string outputFile, mlir::ModuleOp module, CompileOptions &compileOptions)
{
    registerMLIRDialects(module);

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Failed to emit LLVM IR\n";
        return -1;
    }

    if (lldbDebugInfo)
    {
        auto MD = llvmModule->getModuleFlag("CodeView");
        if (llvm::ConstantInt *Behavior = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(MD)) {
            uint64_t Val = Behavior->getLimitedValue();
            if (Val == 1) {
                llvmModule->setModuleFlag(llvm::Module::Warning, "CodeView", static_cast<uint32_t>(0));
            }
        }
    } 

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel, compileOptions);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

#ifndef SAVE_VIA_PASS

    if (emitAction == Action::DumpLLVMIR)
    {
        // TODO: add output into file as well 
        auto FDOut = getOutputStream(emitAction, outputFile);
        if (FDOut)
        {
            FDOut->os() << *llvmModule << "\n";
            FDOut->keep();
        }
        else
        {
            llvm::errs() << *llvmModule << "\n";
        }
    }

    if (emitAction == Action::DumpByteCode)
    {
        auto FDOut = getOutputStream(emitAction, outputFile);
        if (FDOut)
        {
            llvm::WriteBitcodeToFile(*llvmModule, FDOut->os());
            FDOut->keep();
        }
        else
        {
            llvm::WriteBitcodeToFile(*llvmModule, llvm::errs());
        }
    }

#endif

    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module, CompileOptions &compileOptions)
{
    std::string fileOutput = outputFilename.empty() ? getDefaultOutputFileName(emitAction) : outputFilename;
    return dumpLLVMIR(emitAction, fileOutput, module, compileOptions);
}
