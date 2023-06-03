#include "TypeScript/Version.h"
#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRGen.h"
#include "TypeScript/Passes.h"
#include "TypeScript/DiagnosticHelper.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialectTranslation.h"
#include "TypeScript/TypeScriptGC.h"
#ifdef ENABLE_ASYNC
#include "TypeScript/AsyncDialectTranslation.h"
#endif
#ifdef ENABLE_EXCEPTIONS
#include "TypeScript/LandingPadFixPass.h"
#ifdef WIN_EXCEPTION
#include "TypeScript/Win32ExceptionPass.h"
#endif
#endif

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "mlir/Support/DebugCounter.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Host.h"

#ifdef ENABLE_ASYNC
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#endif

#include "llvm/PassInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

// for custom pass
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"

#ifdef GC_ENABLE
#include "llvm/IR/GCStrategy.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#endif

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &, mlir::OwningOpRef<mlir::ModuleOp> &);
int runMLIRPasses(mlir::MLIRContext &, mlir::OwningOpRef<mlir::ModuleOp> &);
int dumpAST();
int dumpLLVMIR(mlir::ModuleOp);
int dumpObjOrAssembly(int, char **, mlir::ModuleOp);
int runJit(int, char **, mlir::ModuleOp);

cl::OptionCategory TypeScriptCompilerCategory("Compiler Options");
cl::OptionCategory TypeScriptCompilerDebugCategory("JIT Debug Options");

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input TypeScript>"), cl::init("-"), cl::value_desc("filename"), cl::cat(TypeScriptCompilerCategory));
cl::opt<std::string> outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"), cl::cat(TypeScriptCompilerCategory));

cl::opt<enum Action> emitAction("emit", cl::desc("Select the kind of output desired"),
                                       cl::values(clEnumValN(DumpAST, "ast", "output AST dump")),
                                       cl::values(clEnumValN(DumpMLIR, "mlir", "output MLIR dump")),
                                       cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output MLIR dump after affine lowering")),
                                       cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output MLIR dump after llvm lowering")),
                                       cl::values(clEnumValN(DumpLLVMIR, "llvm", "output LLVM IR dump")),
                                       cl::values(clEnumValN(DumpByteCode, "bc", "output LLVM ByteCode dump")),
                                       cl::values(clEnumValN(DumpObj, "obj", "output Object file")),
                                       cl::values(clEnumValN(DumpAssembly, "asm", "output LLVM Assembly file")),
                                       cl::values(clEnumValN(RunJIT, "jit", "JIT code and run it by invoking main function")), 
                                       cl::cat(TypeScriptCompilerCategory));

cl::opt<bool> enableOpt{"opt", cl::desc("Enable optimizations"), cl::init(false), cl::cat(TypeScriptCompilerCategory), cl::cat(TypeScriptCompilerCategory)};

cl::opt<int> optLevel{"opt_level", cl::desc("Optimization level"), cl::ZeroOrMore, cl::value_desc("0-3"), cl::init(3), cl::cat(TypeScriptCompilerCategory)};
cl::opt<int> sizeLevel{"size_level", cl::desc("Optimization size level"), cl::ZeroOrMore, cl::value_desc("value"), cl::init(0), cl::cat(TypeScriptCompilerCategory)};

// dump obj
cl::list<std::string> clSharedLibs{"shared-libs", cl::desc("Libraries to link dynamically"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
                                   cl::cat(TypeScriptCompilerCategory)};

cl::opt<std::string> mainFuncName{"e", cl::desc("The function to be called"), cl::value_desc("function name"), cl::init("main"), cl::cat(TypeScriptCompilerCategory)};

cl::opt<bool> dumpObjectFile{"dump-object-file", cl::desc("Dump JITted-compiled object to file specified with "
                                                                 "-object-filename (<input file>.o by default)."), cl::cat(TypeScriptCompilerDebugCategory)};

cl::opt<std::string> objectFilename{"object-filename", cl::desc("Dump JITted-compiled object to file <input file>.o"), cl::cat(TypeScriptCompilerDebugCategory)};

// cl::opt<std::string> targetTriple("mtriple", cl::desc("Override target triple for module"));

cl::opt<bool> disableGC("nogc", cl::desc("Disable Garbage collection"), cl::cat(TypeScriptCompilerCategory));

static void TscPrintVersion(llvm::raw_ostream &OS) {
  OS << "TypeScript Native Compiler (https://github.com/ASDAlexander77/TypeScriptCompiler):" << '\n';
  OS << "  TSNC version " << TSC_PACKAGE_VERSION << '\n' << '\n';

  cl::PrintVersionMessage();
}

int main(int argc, char **argv)
{
    // version printer
    cl::SetVersionPrinter(TscPrintVersion);
    //cl::AddExtraVersionPrinter(TscPrintVersion);

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::DebugCounter::registerCLOptions();

    cl::HideUnrelatedOptions({&TypeScriptCompilerCategory, &TypeScriptCompilerDebugCategory});

    cl::ParseCommandLineOptions(argc, argv, "TypeScript native compiler\n");

    if (emitAction == Action::DumpAST)
    {
        return dumpAST();
    }

    // If we aren't dumping the AST, then we are compiling with/to MLIR.

    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
#ifdef ENABLE_ASYNC
    context.getOrLoadDialect<mlir::async::AsyncDialect>();
#endif

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = compileTypeScriptFileIntoMLIR(context, module))
    {
        return error;
    }

    if (int error = runMLIRPasses(context, module))
    {
        return error;
    }

    // If we aren't exporting to non-mlir, then we are done.
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR)
    {
        module->dump();
        return 0;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR || emitAction == Action::DumpByteCode)
    {
        return dumpLLVMIR(*module);
    }

    if (emitAction == Action::DumpObj || emitAction == Action::DumpAssembly)
    {
        return dumpObjOrAssembly(argc, argv, *module);
    }

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
    {
        return runJit(argc, argv, *module);
    }

    llvm::WithColor::error(llvm::errs(), "tsc") << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
