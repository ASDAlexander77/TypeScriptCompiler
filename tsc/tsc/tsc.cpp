#include "TypeScript/Version.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Debug/Counter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/CodeGen/CommandFlags.h"

// Obj/ASM
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Pass.h"
#include "llvm/InitializePasses.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

CompileOptions prepareOptions();
std::string getDefaultOutputFileName(enum Action);
std::string getDefaultExt(enum Action);
std::string getDefaultLibPath();
int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &, CompileOptions&);
int runMLIRPasses(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &, CompileOptions&);
int dumpAST();
int dumpLLVMIR(mlir::ModuleOp, CompileOptions&);
int dumpObjOrAssembly(int, char **, enum Action, std::string, mlir::ModuleOp, CompileOptions&);
int dumpObjOrAssembly(int, char **, mlir::ModuleOp, CompileOptions&);
int buildExe(int, char **, std::string, CompileOptions&);
int runJit(int, char **, mlir::ModuleOp, CompileOptions&);

extern cl::OptionCategory ObjOrAssemblyCategory;
cl::OptionCategory TypeScriptCompilerCategory("Compiler Options");
cl::OptionCategory TypeScriptCompilerDebugCategory("JIT Debug Options");
cl::OptionCategory TypeScriptCompilerBuildCategory("Executable/Shared library Build Options(used in -emit=exe and -emit=dll)");

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
                                       cl::init(RunJIT),
                                       cl::cat(TypeScriptCompilerCategory));

cl::opt<bool> enableOpt{"opt", cl::desc("Enable optimizations"), cl::init(false), cl::cat(TypeScriptCompilerCategory), cl::cat(TypeScriptCompilerCategory)};

cl::opt<int> optLevel{"opt_level", cl::desc("Optimization level"), cl::ZeroOrMore, cl::value_desc("0-3"), cl::init(3), cl::cat(TypeScriptCompilerCategory)};
cl::opt<int> sizeLevel{"size_level", cl::desc("Optimization size level"), cl::ZeroOrMore, cl::value_desc("value"), cl::init(0), cl::cat(TypeScriptCompilerCategory)};

// dump obj
cl::list<std::string> clSharedLibs{"shared-libs", cl::desc("Libraries to link dynamically. (used in --emit=jit)"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
                                   cl::cat(TypeScriptCompilerCategory)};

cl::opt<std::string> mainFuncName{"e", cl::desc("The function to be called (default=main)"), cl::value_desc("function name"), cl::init("main"), cl::cat(TypeScriptCompilerCategory)};

cl::opt<bool> dumpObjectFile{"dump-object-file", cl::Hidden, cl::desc("Dump JITted-compiled object to file specified with "
        "-object-filename (<input file>.o by default)."), cl::init(false), cl::cat(TypeScriptCompilerDebugCategory)};

cl::opt<std::string> objectFilename{"object-filename", cl::Hidden, cl::desc("Dump JITted-compiled object to file <input file>.o"), cl::cat(TypeScriptCompilerDebugCategory)};

cl::opt<bool> verbose{"verbose", cl::Hidden, cl::desc("Verbose output"), cl::init(false), cl::cat(TypeScriptCompilerDebugCategory)};
cl::opt<bool> printOp{"print-op", cl::Hidden, cl::desc("Print Op on Diagnostic"), cl::init(false), cl::cat(TypeScriptCompilerDebugCategory)};
cl::opt<bool> printStackTrace{"print-stack-trace", cl::Hidden, cl::desc("Print stack trace on Diagnostic"), cl::init(false), cl::cat(TypeScriptCompilerDebugCategory)};

// cl::opt<std::string> targetTriple("mtriple", cl::desc("Override target triple for module"));

cl::opt<bool> disableGC("nogc", cl::desc("Disable Garbage collection"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> disableWarnings("nowarn", cl::desc("Disable Warnings"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> generateDebugInfo("di", cl::desc("Generate Debug Infomation"), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> lldbDebugInfo("lldb", cl::desc("Debug Infomation for LLDB"), cl::cat(TypeScriptCompilerCategory));
cl::opt<enum Exports> exportAction("export", cl::desc("Export Symbols. (Useful to compile the same code into 'lib' (static library) and/or 'dll/so' (dynamic library)) "),
                                       cl::values(clEnumValN(ExportAll, "all", "export all symbols")),
                                       cl::values(clEnumValN(IgnoreAll, "none", "ignore all exports")),
                                       cl::cat(TypeScriptCompilerCategory));

cl::opt<std::string> defaultlibpath("default-lib-path", cl::desc("JS library path. Should point to folder/directory with subfolder '" DEFAULT_LIB_DIR "' or DEFAULT_LIB_PATH environmental variable"), cl::value_desc("defaultlibpath"), cl::cat(TypeScriptCompilerBuildCategory));
cl::opt<std::string> gclibpath("gc-lib-path", cl::desc("GC library path. Should point to file 'gcmt-lib.lib' or GC_LIB_PATH environmental variable"), cl::value_desc("gclibpath"), cl::cat(TypeScriptCompilerBuildCategory));
cl::opt<std::string> llvmlibpath("llvm-lib-path", cl::desc("LLVM library path. Should point to file 'LLVMSupport.lib' and 'LLVMDemangle' in linux or LLVM_LIB_PATH environmental variable"), cl::value_desc("llvmlibpath"), cl::cat(TypeScriptCompilerBuildCategory));
cl::opt<std::string> tsclibpath("tsc-lib-path", cl::desc("TypeScript Compiler Runtime library path. Should point to file 'TypeScriptAsyncRuntime.lib' or TSC_LIB_PATH environmental variable"), cl::value_desc("tsclibpath"), cl::cat(TypeScriptCompilerBuildCategory));
cl::opt<std::string> emsdksysrootpath("emsdk-sysroot-path", cl::desc("TypeScript Compiler Runtime library path. Should point to dir '<...>/emsdk/upstream/emscripten/cache/sysroot' or EMSDK_SYSROOT_PATH environmental variable. (used when '-mtriple=wasm32-pc-emscripten')"), cl::value_desc("emsdksysrootpath"), cl::cat(TypeScriptCompilerBuildCategory));
cl::list<std::string> libs{"lib", cl::desc("Libraries to link statically. (used in --emit=exe)"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated, cl::cat(TypeScriptCompilerBuildCategory)};
cl::list<std::string> objs{"obj", cl::desc("Object files to link statically. (used in --emit=exe and --emit=dll)"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated, cl::cat(TypeScriptCompilerBuildCategory)};

cl::opt<bool> noDefaultLib("no-default-lib", cl::desc("Disable loading default lib"), cl::init(false), cl::cat(TypeScriptCompilerCategory));
cl::opt<bool> enableBuiltins("builtins", cl::desc("Builtin functionality (needed if Default lib is not provided)"), cl::init(true), cl::cat(TypeScriptCompilerCategory));

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

std::string GetTemporaryPath(llvm::StringRef Prefix, llvm::StringRef Suffix)
{
    llvm::SmallString<256> Path;
    auto EC = llvm::sys::fs::createTemporaryFile(Prefix, Suffix, Path);
    if (EC)
    {
        return "";
    }

    return std::string(Path.str());
}

std::string mergeWithDefaultLibPath(std::string defaultlibpath, std::string subPath)
{
    if (defaultlibpath.empty())
    {
        return subPath;
    }

    llvm::SmallVector<char> path(0);
    llvm::SmallVector<char> nativePath(0);
    llvm::sys::path::append(path, defaultlibpath, subPath);
    llvm::sys::path::native(path, nativePath);    
    llvm::StringRef str(nativePath.data(), nativePath.size());
    return str.str();
}

bool prepareDefaultLib(CompileOptions &compileOptions)
{
    if (noDefaultLib)
    {
        return true;
    }

    // TODO: temp hack
    auto defaultLibPathVariable = getDefaultLibPath();
    auto fullPath = mergeWithDefaultLibPath(defaultLibPathVariable, DEFAULT_LIB_DIR "/");
    auto isDir = llvm::sys::fs::is_directory(fullPath);
    if (!defaultLibPathVariable.empty() && !isDir) 
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Default lib path: " << fullPath
                                    << " does not exist or is not a directory\n";
        return false;
    }

    compileOptions.noDefaultLib |= !isDir;
    if (!compileOptions.noDefaultLib)
    {
        compileOptions.defaultDeclarationTSFile = mergeWithDefaultLibPath(getDefaultLibPath(), DEFAULT_LIB_DIR "/lib.d.ts");
    }

    return true;
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
    mlir::tracing::DebugCounter::registerCLOptions();

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
    llvm::initializeHardwareLoopsLegacyPass(*Registry);
    llvm::initializeTransformUtils(*Registry);
    llvm::initializeReplaceWithVeclibLegacyPass(*Registry);
    llvm::initializeTLSVariableHoistLegacyPassPass(*Registry);

    // Initialize debugging passes.
    llvm::initializeScavengerTestPass(*Registry);
    
    // End - Register for Obj/ASM

    cl::HideUnrelatedOptions({&TypeScriptCompilerCategory, &TypeScriptCompilerDebugCategory, &TypeScriptCompilerBuildCategory, &ObjOrAssemblyCategory});
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
    mlir::DialectRegistry registry;
    //mlir::func::registerAllExtensions(registry);
    registerAllExtensions(registry);

    mlir::MLIRContext mlirContext(registry);
    // Load our Dialect in this MLIR Context.
    mlirContext.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    mlirContext.getOrLoadDialect<mlir::arith::ArithDialect>();
    mlirContext.getOrLoadDialect<mlir::math::MathDialect>();
    mlirContext.getOrLoadDialect<mlir::index::IndexDialect>();
    mlirContext.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    mlirContext.getOrLoadDialect<mlir::func::FuncDialect>();
    mlirContext.getOrLoadDialect<mlir::DLTIDialect>();
    mlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
#ifdef ENABLE_ASYNC
    mlirContext.getOrLoadDialect<mlir::async::AsyncDialect>();
#endif

    mlirContext.printOpOnDiagnostic(printOp.getValue());
    mlirContext.printStackTraceOnDiagnostic(printStackTrace.getValue());

    auto compileOptions = prepareOptions();

    if (!prepareDefaultLib(compileOptions))
    {
        return 0;
    }

    llvm::SourceMgr sourceMgr;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = compileTypeScriptFileIntoMLIR(mlirContext, sourceMgr, module, compileOptions))
    {
        return error;
    }

    if (int error = runMLIRPasses(mlirContext, sourceMgr, module, compileOptions))
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
        return dumpLLVMIR(*module, compileOptions);
    }

    if (emitAction == Action::DumpObj || emitAction == Action::DumpAssembly)
    {
        return dumpObjOrAssembly(argc, argv, *module, compileOptions);
    }

    if (emitAction == Action::BuildExe || emitAction == Action::BuildDll)
    {
        auto defaultFilePath = getDefaultOutputFileName(Action::DumpObj);
        auto fileName = llvm::sys::path::stem(defaultFilePath);
        auto ext = getDefaultExt(Action::DumpObj);
        auto tempOutputFile = GetTemporaryPath(fileName, ext);
        auto result = dumpObjOrAssembly(argc, argv, Action::DumpObj, tempOutputFile, *module, compileOptions);
        if (result != 0)
        {
            return result;
        }

        return buildExe(argc, argv, tempOutputFile, compileOptions);
    }

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
    {
        return runJit(argc, argv, *module, compileOptions);
    }

    llvm::WithColor::error(llvm::errs(), "tsc") << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
