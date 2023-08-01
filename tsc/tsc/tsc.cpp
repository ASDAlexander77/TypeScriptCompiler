#include "TypeScript/Version.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugCounter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/CodeGen/CommandFlags.h"

// Obj/ASM
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Pass.h"
#include "llvm/InitializePasses.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

std::string getDefaultOutputFileName(enum Action);
int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &);
int runMLIRPasses(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &);
int dumpAST();
int dumpLLVMIR(mlir::ModuleOp);
int dumpObjOrAssembly(int, char **, enum Action, std::string, mlir::ModuleOp);
int dumpObjOrAssembly(int, char **, mlir::ModuleOp);
int buildExe(int, char **, std::string);
int runJit(int, char **, mlir::ModuleOp);

extern cl::OptionCategory ObjOrAssemblyCategory;
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
#ifdef WIN32                                       
                                       cl::values(clEnumValN(BuildExe, "exe", "build Executable (.exe) file")),
#else                                       
                                       cl::values(clEnumValN(BuildExe, "exe", "build Executable file")),
#endif
#ifdef WIN32                                       
                                       cl::values(clEnumValN(BuildDll, "dll", "build Dynamic Link Library (.dll) file")),
#else                                       
                                       cl::values(clEnumValN(BuildDll, "dll", "build Shared library (.so/.dylib) file")),
#endif
                                       cl::values(clEnumValN(RunJIT, "jit", "JIT code and run it by invoking main function")), 
                                       cl::cat(TypeScriptCompilerCategory));

cl::opt<bool> enableOpt{"opt", cl::desc("Enable optimizations"), cl::init(false), cl::cat(TypeScriptCompilerCategory), cl::cat(TypeScriptCompilerCategory)};

cl::opt<int> optLevel{"opt_level", cl::desc("Optimization level"), cl::ZeroOrMore, cl::value_desc("0-3"), cl::init(3), cl::cat(TypeScriptCompilerCategory)};
cl::opt<int> sizeLevel{"size_level", cl::desc("Optimization size level"), cl::ZeroOrMore, cl::value_desc("value"), cl::init(0), cl::cat(TypeScriptCompilerCategory)};

// dump obj
cl::list<std::string> clSharedLibs{"shared-libs", cl::desc("Libraries to link dynamically"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
                                   cl::cat(TypeScriptCompilerCategory)};

cl::opt<std::string> mainFuncName{"e", cl::desc("The function to be called (default=main)"), cl::value_desc("function name"), cl::init("main"), cl::cat(TypeScriptCompilerCategory)};

cl::opt<bool> dumpObjectFile{"dump-object-file", cl::Hidden, cl::desc("Dump JITted-compiled object to file specified with "
                                                                 "-object-filename (<input file>.o by default)."), cl::cat(TypeScriptCompilerDebugCategory)};

cl::opt<std::string> objectFilename{"object-filename", cl::Hidden, cl::desc("Dump JITted-compiled object to file <input file>.o"), cl::cat(TypeScriptCompilerDebugCategory)};

// cl::opt<std::string> targetTriple("mtriple", cl::desc("Override target triple for module"));

cl::opt<bool> disableGC("nogc", cl::desc("Disable Garbage collection"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> disableWarnings("nowarn", cl::desc("Disable Warnings"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> generateDebugInfo("di", cl::desc("Generate Debug Infomation"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> lldbDebugInfo("lldb", cl::desc("Debug Infomation for LLDB"), cl::cat(TypeScriptCompilerCategory));

static void TscPrintVersion(llvm::raw_ostream &OS) {
  OS << "TypeScript Native Compiler (https://github.com/ASDAlexander77/TypeScriptCompiler):" << '\n';
  OS << "  TSNC version " << TSC_PACKAGE_VERSION << '\n' << '\n';

  cl::PrintVersionMessage();
}

void HideUnrelatedOptionsButVisibleForHidden(cl::SubCommand &Sub) {
    for (auto &I : Sub.OptionsMap) {
        if (I.second->getOptionHiddenFlag() == cl::ReallyHidden)
            I.second->setHiddenFlag(cl::Hidden/*cl::ReallyHidden*/);
    }
}

int main(int argc, char **argv)
{
    // version printer
    cl::SetVersionPrinter(TscPrintVersion);

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::DebugCounter::registerCLOptions();

    // Register for Obj/ASM
    // Initialize targets first, so that --version shows registered targets.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    // Initialize codegen and IR passes used by llc so that the -print-after,
    // -print-before, and -stop-after options work.
    llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(*Registry);
    llvm::initializeCodeGen(*Registry);
    llvm::initializeLoopStrengthReducePass(*Registry);
    llvm::initializeLowerIntrinsicsPass(*Registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
    llvm::initializeConstantHoistingLegacyPassPass(*Registry);
    llvm::initializeScalarOpts(*Registry);
    llvm::initializeVectorization(*Registry);
    llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
    llvm::initializeExpandReductionsPass(*Registry);
    llvm::initializeExpandVectorPredicationPass(*Registry);
    llvm::initializeHardwareLoopsPass(*Registry);
    llvm::initializeTransformUtils(*Registry);
    llvm::initializeReplaceWithVeclibLegacyPass(*Registry);
    llvm::initializeTLSVariableHoistLegacyPassPass(*Registry);

    // Initialize debugging passes.
    llvm::initializeScavengerTestPass(*Registry);
    
    // End - Register for Obj/ASM

    cl::HideUnrelatedOptions({&TypeScriptCompilerCategory, &TypeScriptCompilerDebugCategory, &ObjOrAssemblyCategory});
    HideUnrelatedOptionsButVisibleForHidden(SubCommand::getTopLevel());

    // Register the Target and CPU printer for --version.
    cl::AddExtraVersionPrinter(llvm::sys::printDefaultTargetAndDetectedCPU);
    // Register the target printer for --version.
    cl::AddExtraVersionPrinter(llvm::TargetRegistry::printRegisteredTargetsForVersion);

    cl::ParseCommandLineOptions(argc, argv, "TypeScript native compiler\n");

    if (emitAction == Action::DumpAST)
    {
        return dumpAST();
    }

    // If we aren't dumping the AST, then we are compiling with/to MLIR.

    mlir::MLIRContext mlirContext;
    // Load our Dialect in this MLIR Context.
    mlirContext.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    mlirContext.getOrLoadDialect<mlir::arith::ArithDialect>();
    mlirContext.getOrLoadDialect<mlir::math::MathDialect>();
    mlirContext.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    mlirContext.getOrLoadDialect<mlir::func::FuncDialect>();
    mlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
#ifdef ENABLE_ASYNC
    mlirContext.getOrLoadDialect<mlir::async::AsyncDialect>();
#endif

#ifdef NDEBUG
    mlirContext.printOpOnDiagnostic(false);
#else 
    mlirContext.printStackTraceOnDiagnostic(true);
#endif

    mlir::OwningOpRef<mlir::ModuleOp> module;

    llvm::SourceMgr sourceMgr;
    if (int error = compileTypeScriptFileIntoMLIR(mlirContext, sourceMgr, module))
    {
        return error;
    }

    if (int error = runMLIRPasses(mlirContext, sourceMgr, module))
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

    if (emitAction == Action::BuildExe || emitAction == Action::BuildDll)
    {
        auto tempOutputFile = getDefaultOutputFileName(Action::DumpObj);
        auto result = dumpObjOrAssembly(argc, argv, Action::DumpObj, tempOutputFile, *module);
        if (result != 0)
        {
            return result;
        }

        return buildExe(argc, argv, tempOutputFile);
    }

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
    {
        return runJit(argc, argv, *module);
    }

    llvm::WithColor::error(llvm::errs(), "tsc") << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
