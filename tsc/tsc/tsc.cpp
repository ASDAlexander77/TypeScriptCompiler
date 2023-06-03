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
//#include "mlir/Support/ToolUtilities.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/FileUtilities.h"
//#include "llvm/Support/Regex.h"
//#include "llvm/Support/SourceMgr.h"
//#include "llvm/Support/StringSaver.h"
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

// for dump obj
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

// end of dump obj

#define DEBUG_TYPE "tsc"

#define ENABLE_CUSTOM_PASSES 1
#define ENABLE_OPT_PASSES 1
//#define SAVE_VIA_PASS 1
// TODO: if you uncomment it you will have exception in test 00try_finally.ts error: empty block: expect at least a terminator
//#define AFFINE_MODULE_PASS 1

using namespace typescript;
namespace cl = llvm::cl;

int runJit(int, char **, mlir::ModuleOp);

// obj
static llvm::codegen::RegisterCodeGenFlags CGF;

cl::OptionCategory TypeScriptCompilerCategory("Compiler Options");
cl::OptionCategory TypeScriptCompilerDebugCategory("JIT Debug Options");

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input TypeScript>"), cl::init("-"), cl::value_desc("filename"), cl::cat(TypeScriptCompilerCategory));
cl::opt<std::string> outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"), cl::cat(TypeScriptCompilerCategory));

namespace
{
enum Action
{
    None,
    DumpAST,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    DumpByteCode,
    DumpObj,
    DumpAssembly,
    RunJIT
};
} // namespace

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

int runMLIRPasses(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp>  &module)
{
    mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedMessages;
    mlir::ScopedDiagnosticHandler diagHandler(&context, [&](mlir::Diagnostic &diag) {
        postponedMessages.emplace_back(new mlir::Diagnostic(std::move(diag)));
    });

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (isLoweringToAffine)
    {
        pm.addPass(mlir::createCanonicalizerPass());

#ifdef ENABLE_ASYNC
        pm.addPass(mlir::createAsyncToAsyncRuntimePass());
#endif

#ifndef AFFINE_MODULE_PASS
        mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM.addPass(mlir::typescript::createLowerToAffineTSFuncPass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::typescript::createRelocateConstantPass());

        mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM2.addPass(mlir::typescript::createLowerToAffineFuncPass());
        optPM2.addPass(mlir::createCanonicalizerPass());

        pm.addPass(mlir::typescript::createLowerToAffineModulePass());
        pm.addPass(mlir::createCanonicalizerPass());
#else        
        pm.addPass(mlir::typescript::createLowerToAffineModulePass());
        pm.addPass(mlir::createCanonicalizerPass());

        mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();
        optPM.addPass(mlir::typescript::createRelocateConstantPass());
#endif

#ifdef ENABLE_OPT_PASSES
        if (enableOpt)
        {
            optPM.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createLoopInvariantCodeMotionPass());
            pm.addPass(mlir::createStripDebugInfoPass());
            pm.addPass(mlir::createInlinerPass());
            pm.addPass(mlir::createSCCPPass());
            pm.addPass(mlir::createSymbolDCEPass());
        }
#endif

#ifdef ENABLE_ASYNC
        // pm.addPass(mlir::createAsyncToAsyncRuntimePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createAsyncRuntimeRefCountingPass());
        if (enableOpt)
        {
            pm.addPass(mlir::createAsyncRuntimeRefCountingOptPass());
        }
#endif
    }

    if (isLoweringToLLVM)
    {
#ifdef ENABLE_ASYNC
        pm.addPass(mlir::createConvertAsyncToLLVMPass());
#endif
        pm.addPass(mlir::typescript::createLowerToLLVMPass());
        if (!disableGC)
        {
            pm.addPass(mlir::typescript::createGCPass());
        }
    }

    auto result = 0;
    if (mlir::failed(pm.run(*module)))
    {
        result = 4;
    }

    printDiagnostics(postponedMessages);
    return result;
}

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

int registerMLIRDialects(mlir::ModuleOp module)
{
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*module->getContext());
    mlir::typescript::registerTypeScriptDialectTranslation(*module->getContext());

#ifdef TSGC_ENABLE
    mlir::typescript::registerTypeScriptGC();
#endif

#ifdef ENABLE_ASYNC
    mlir::typescript::registerAsyncDialectTranslation(*module->getContext());
#endif

    return 0;
}

static llvm::Optional<llvm::OptimizationLevel> mapToLevel(unsigned optLevel, unsigned sizeLevel)
{
    switch (optLevel)
    {
    case 0:
        return llvm::OptimizationLevel::O0;

    case 1:
        return llvm::OptimizationLevel::O1;

    case 2:
        switch (sizeLevel)
        {
        case 0:
            return llvm::OptimizationLevel::O2;

        case 1:
            return llvm::OptimizationLevel::Os;

        case 2:
            return llvm::OptimizationLevel::Oz;
        }
        break;
    case 3:
        return llvm::OptimizationLevel::O3;
    }
    return std::nullopt;
}

static char mapToLevel(unsigned optLevel)
{
    switch (optLevel)
    {
    case 0:
        return '0';
    case 1:
        return '1';
    case 2:
        return '2';
    case 3:
        return '3';
    }

    return '0';
}


static std::unique_ptr<llvm::ToolOutputFile> getOutputStream()
{
    // If we don't yet have an output filename, make one.
    if (outputFilename.empty())
    {
        if (inputFilename == "-")
            outputFilename = "-";
        else
        {
            // If InputFilename ends in .bc or .ll, remove it.
            llvm::StringRef IFN = inputFilename;
            if (IFN.endswith(".ts"))
                outputFilename = std::string(IFN.drop_back(3));
            else if (IFN.endswith(".mlir"))
                outputFilename = std::string(IFN.drop_back(5));
            else
                outputFilename = std::string(IFN);

            switch (emitAction)
            {
                case None:
                    outputFilename = "-";
                    break;
                case DumpAST:
                    outputFilename += ".txt";
                    break;
                case DumpMLIR:
                case DumpMLIRAffine:
                case DumpMLIRLLVM:
                    outputFilename += ".mlir";
                    break;
                case DumpLLVMIR:
                    outputFilename += ".ll";
                    break;
                case DumpByteCode:
                    outputFilename += ".bc";
                    break;
                case DumpObj:
                    {
                        llvm::Triple theTriple;
                        theTriple.setTriple(llvm::sys::getDefaultTargetTriple());
                        outputFilename += (theTriple.getOS() == llvm::Triple::Win32) ? ".obj" : ".o";
                    }

                    break;
                case DumpAssembly:
                    outputFilename += ".asm";
                    break;
                case RunJIT:
                    outputFilename = "-";
                    break;
            }
        }
    }

    // Open the file.
    std::error_code EC;
    llvm::sys::fs::OpenFlags openFlags = llvm::sys::fs::OF_None;
    if (emitAction != DumpByteCode && emitAction != DumpObj)
    {
        openFlags |= llvm::sys::fs::OF_TextWithCRLF;
    }

    auto FDOut = std::make_unique<llvm::ToolOutputFile>(outputFilename, EC, openFlags);
    if (EC)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << EC.message() << "\n";
        return nullptr;
    }

    return FDOut;
}

std::function<llvm::Error(llvm::Module *)> makeCustomPassesWithOptimizingTransformer(
        llvm::Optional<unsigned> mbOptLevel, llvm::Optional<unsigned> mbSizeLevel, llvm::TargetMachine *targetMachine)
{
    return [mbOptLevel, mbSizeLevel, targetMachine](llvm::Module *m) -> llvm::Error {
        llvm::Optional<llvm::OptimizationLevel> ol = mapToLevel(mbOptLevel.value(), mbSizeLevel.value());
        if (!ol)
        {
            return llvm::make_error<llvm::StringError>(
                llvm::formatv("invalid optimization/size level {0}/{1}", mbOptLevel.value(), mbSizeLevel.value()).str(),
                llvm::inconvertibleErrorCode());
        }
        
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;

        llvm::PassBuilder pb(targetMachine);

        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;

        //pb.parsePassPipeline(mpm, "module(function(landing-pad-fix))");

        // add custom passes
        mpm.addPass(llvm::createModuleToFunctionPassAdaptor(ts::LandingPadFixPass()));
#ifdef WIN_EXCEPTION        
        mpm.addPass(llvm::createModuleToFunctionPassAdaptor(ts::Win32ExceptionPass()));
#endif

        if (*ol == llvm::OptimizationLevel::O0)
            mpm.addPass(pb.buildO0DefaultPipeline(*ol));
        else
            mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));


#ifdef SAVE_VIA_PASS
        std::unique_ptr<llvm::ToolOutputFile> FDOut;
        if (emitAction == Action::DumpLLVMIR)
        {
            FDOut = GetOutputStream();
            mpm.addPass(llvm::PrintModulePass(FDOut ? FDOut->os() : llvm::errs()));
        }

        if (emitAction == Action::DumpByteCode)
        {
            FDOut = GetOutputStream();
            mpm.addPass(llvm::BitcodeWriterPass(FDOut ? FDOut->os() : llvm::errs()));
        }
#endif        

        mpm.run(*m, mam);

#ifdef SAVE_VIA_PASS
        if (FDOut)
        {
            FDOut->keep();
        }
#endif        

        return llvm::Error::success();
    };
}

std::function<llvm::Error(llvm::Module *)> getTransformer(bool enableOpt, int optLevel, int sizeLevel)
{
#ifdef ENABLE_CUSTOM_PASSES
    auto optPipeline = makeCustomPassesWithOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0, 
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        /*targetMachine=*/nullptr);
#else
    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0, 
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        /*targetMachine=*/nullptr);
#endif

    return optPipeline;
}

int dumpLLVMIR(mlir::ModuleOp module)
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

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

#ifndef SAVE_VIA_PASS

    if (emitAction == Action::DumpLLVMIR)
    {
        // TODO: add output into file as well 
        auto FDOut = getOutputStream();
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
        auto FDOut = getOutputStream();
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

struct LLCDiagnosticHandler : public llvm::DiagnosticHandler {
  bool *HasError;
  LLCDiagnosticHandler(bool *HasErrorPtr) : HasError(HasErrorPtr) {}
  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
    if (DI.getKind() == llvm::DK_SrcMgr) {
      const auto &DISM = llvm::cast<llvm::DiagnosticInfoSrcMgr>(DI);
      const llvm::SMDiagnostic &SMD = DISM.getSMDiag();

      if (SMD.getKind() == llvm::SourceMgr::DK_Error)
        *HasError = true;

      SMD.print(nullptr, llvm::errs());

      // For testing purposes, we print the LocCookie here.
      if (DISM.isInlineAsmDiag() && DISM.getLocCookie())
        llvm::WithColor::note() << "!srcloc = " << DISM.getLocCookie() << "\n";

      return true;
    }

    if (DI.getSeverity() == llvm::DS_Error)
      *HasError = true;

    if (auto *Remark = llvm::dyn_cast<llvm::DiagnosticInfoOptimizationBase>(&DI))
      if (!Remark->isEnabled())
        return true;

    llvm::DiagnosticPrinterRawOStream DP(llvm::errs());
    llvm::errs() << llvm::LLVMContext::getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
    DI.print(DP);
    llvm::errs() << "\n";
    return true;
  }
};

int dumpObjOrAssembly(int argc, char **argv, mlir::ModuleOp module)
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

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    //
    // generate Obj
    //
    cl::ParseCommandLineOptions(argc, argv, "tsc\n");

    llvm::LLVMContext Context;
    // set from command line opt
    //Context.setDiscardValueNames(DiscardValueNames);

    // Set a diagnostic handler that doesn't exit on the first error
    bool HasError = false;
    Context.setDiagnosticHandler(std::make_unique<LLCDiagnosticHandler>(&HasError));

    // set options
    llvm::TargetOptions Options;
    auto InitializeOptions = [&](const llvm::Triple &TheTriple) {
        Options = llvm::codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
        // Options.BinutilsVersion = llvm::TargetMachine::parseBinutilsVersion(BinutilsVersion);
        // Options.DisableIntegratedAS = NoIntegratedAssembler;
        // Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
        // Options.MCOptions.AsmVerbose = AsmVerbose;
        // Options.MCOptions.PreserveAsmComments = PreserveComments;
        // Options.MCOptions.IASSearchPaths = IncludeDirs;
        // Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
        // if (DwarfDirectory.getPosition()) 
        // {
        //     Options.MCOptions.MCUseDwarfDirectory =
        //     DwarfDirectory ? llvm::MCTargetOptions::EnableDwarfDirectory
        //                    : llvm::MCTargetOptions::DisableDwarfDirectory;
        // } else {
        //     // -dwarf-directory is not set explicitly. Some assemblers
        //     // (e.g. GNU as or ptxas) do not support `.file directory'
        //     // syntax prior to DWARFv5. Let the target decide the default
        //     // value.
        //     Options.MCOptions.MCUseDwarfDirectory = MCTargetOptions::DefaultDwarfDirectory;
        // }
    };

    std::optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();
    std::optional<llvm::CodeModel::Model> CM = llvm::codegen::getExplicitCodeModel();

    llvm::Triple TheTriple;
    const llvm::Target *TheTarget = nullptr;
    std::unique_ptr<llvm::TargetMachine> Target;
    
    // If we are supposed to override the target triple, do so now.
    std::string LLVMModuleTargetTriple = llvmModule.get()->getTargetTriple();
    /*
    // override from command line here
    */
    
    TheTriple = llvm::Triple(LLVMModuleTargetTriple);
    if (TheTriple.getTriple().empty())
    {
        TheTriple.setTriple(llvm::sys::getDefaultTargetTriple());
    }

    std::string Error;
    TheTarget = llvm::TargetRegistry::lookupTarget(llvm::codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) 
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << Error;
        return -1;
    }

      // On AIX, setting the relocation model to anything other than PIC is
      // considered a user error.
    if (TheTriple.isOSAIX() && RM && *RM != llvm::Reloc::PIC_)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "invalid relocation model, AIX only supports PIC";
        return -1;
    }

    auto CPUStr = llvm::codegen::getCPUStr(), 
         FeaturesStr = llvm::codegen::getFeaturesStr();

    llvm::CodeGenOpt::Level OLvl;
    // TODO: finish it, optimization
    if (auto Level = llvm::CodeGenOpt::parseLevel(enableOpt ? mapToLevel(optLevel) : '0')) 
    {
        OLvl = *Level;
    } 
    else 
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "invalid optimization level.\n";
        return 1;
    }


    InitializeOptions(TheTriple);
    Target = std::unique_ptr<llvm::TargetMachine>(TheTarget->createTargetMachine(
          TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");

    assert(llvmModule.get() && "Should have exited if we didn't have a module!");
    if (llvm::codegen::getFloatABIForCalls() != llvm::FloatABI::Default)
    {
        Options.FloatABIType = llvm::codegen::getFloatABIForCalls();
    }

    auto FDOut = getOutputStream();
    if (!FDOut)
    {
        return -1;
    }

    // Ensure the filename is passed down to CodeViewDebug.
    Target->Options.ObjectFilenameForDebug = FDOut->outputFilename();

    // Build up all of the passes that we want to do to the module.
    llvm::legacy::PassManager PM;

    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(llvmModule->getTargetTriple()));

    PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

#ifndef NDEBUG
    // TODO: disable it in release
    if (llvm::verifyModule(*llvmModule.get(), &llvm::errs()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "input module cannot be verified\n";
        return -1;        
    }
#endif    

    // Override function attributes based on CPUStr, FeaturesStr, and command line flags.
    llvm::codegen::setFunctionAttributes(CPUStr, FeaturesStr, *llvmModule.get());

    auto fileFormat = emitAction == DumpObj ? llvm::CGFT_ObjectFile : emitAction == DumpAssembly ? llvm::CGFT_AssemblyFile : llvm::CGFT_Null;

    if (llvm::mc::getExplicitRelaxAll() && /*llvm::codegen::getFileType()*/ fileFormat != llvm::CGFT_ObjectFile)
    {
        llvm::WithColor::warning(llvm::errs(), "tsc") << ": warning: ignoring -mc-relax-all because filetype != obj";
    }

    {
        llvm::raw_pwrite_stream *OS = &FDOut->os();

        // Manually do the buffering rather than using buffer_ostream,
        // so we can memcmp the contents in CompileTwice mode
        llvm::SmallVector<char, 0> Buffer;
        std::unique_ptr<llvm::raw_svector_ostream> BOS;
        if ((/*llvm::codegen::getFileType()*/ fileFormat != llvm::CGFT_AssemblyFile &&
            !FDOut->os().supportsSeeking())) 
        {
            BOS = std::make_unique<llvm::raw_svector_ostream>(Buffer);
            OS = BOS.get();
        }

        auto &LLVMTM = static_cast<llvm::LLVMTargetMachine &>(*Target);
        auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM);

        if (Target->addPassesToEmitFile(
                        PM, *OS, /*DwoOut ? &DwoOut->os() : */nullptr,
                        fileFormat /*llvm::codegen::getFileType()*/, true, MMIWP)) 
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "target does not support generation of this file type\n";
        }

        const_cast<llvm::TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())->Initialize(MMIWP->getMMI().getContext(), *Target);

        // Before executing passes, print the final values of the LLVM options.
        llvm::cl::PrintOptionValues();

        PM.run(*llvmModule.get());

        auto HasError = ((const LLCDiagnosticHandler *)(Context.getDiagHandlerPtr()))->HasError;
        if (*HasError)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "diagnostic error(s)\n";
            return 1;
        }

        if (BOS) 
        {
            FDOut->os() << Buffer;
        }
    }

    // Declare success.
    FDOut->keep();

    return 0;
}

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
