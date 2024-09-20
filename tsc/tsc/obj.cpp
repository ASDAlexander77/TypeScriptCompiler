#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

// for dump obj
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/DataStructs.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::opt<std::string> outputFilename;

std::string getDefaultOutputFileName(enum Action);
std::unique_ptr<llvm::ToolOutputFile> getOutputStream(enum Action, std::string);
int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int, CompileOptions&);

static llvm::codegen::RegisterCodeGenFlags CGF;

// Cmdline options
cl::OptionCategory ObjOrAssemblyCategory("OBJ/Assembly Options");

static cl::opt<bool> 
    DiscardValueNames("discard-value-names",
                    cl::desc("Discard names from Value (other than GlobalValue)."),
                    cl::init(false), cl::Hidden, cl::cat(ObjOrAssemblyCategory));

static cl::list<std::string> IncludeDirs("I", cl::desc("include search path"), cl::cat(ObjOrAssemblyCategory));

static cl::opt<std::string>
    BinutilsVersion("binutils-version", cl::Hidden,
                    cl::desc("Produced object files can use all ELF features "
                             "supported by this binutils version and newer."
                             "If -no-integrated-as is specified, the generated "
                             "assembly will consider GNU as support."
                             "'none' means that all ELF features can be used, "
                             "regardless of binutils support"), cl::cat(ObjOrAssemblyCategory));

// TODO: find out why we can't add our 1
// static cl::opt<bool>
//     NoIntegratedAssembler("no-integrated-as", cl::Hidden,
//                       cl::desc("Disable integrated assembler"), cl::cat(ObjOrAssemblyCategory));

static cl::opt<bool>
    PreserveComments("preserve-as-comments", cl::Hidden,
                     cl::desc("Preserve Comments in outputted assembly"),
                     cl::init(true), cl::cat(ObjOrAssemblyCategory));

cl::opt<std::string> TargetTriple("mtriple", 
                                  cl::desc("Override target triple for module"), 
                                  cl::cat(ObjOrAssemblyCategory));

static cl::opt<std::string> SplitDwarfFile("split-dwarf-file", 
                                            cl::desc("Specify the name of the .dwo file to encode in the DWARF output"), cl::cat(ObjOrAssemblyCategory));

static cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                              cl::desc("Do not verify input module"), cl::cat(ObjOrAssemblyCategory));

static cl::opt<bool> ShowMCEncoding("show-mc-encoding", cl::Hidden,
                                    cl::desc("Show encoding in .s output"), cl::cat(ObjOrAssemblyCategory));

static cl::opt<std::string> SplitDwarfOutputFile("split-dwarf-output", cl::Hidden,
                                                 cl::desc(".dwo output filename"), cl::value_desc("filename"));

static cl::opt<bool> DwarfDirectory("dwarf-directory", cl::Hidden,
                                    cl::desc("Use .file directives with an explicit directory"),
                                    cl::init(true), cl::cat(ObjOrAssemblyCategory));

static cl::opt<bool> AsmVerbose("asm-verbose",
                                cl::desc("Add comments to directives."),
                                cl::init(true), cl::cat(ObjOrAssemblyCategory));


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

int setupTargetTriple(llvm::Module *llvmModule, std::unique_ptr<llvm::TargetMachine> &Target, llvm::TargetOptions &Options)
{
    auto InitializeOptions = [&](const llvm::Triple &TheTriple) {
        Options = llvm::codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
        Options.BinutilsVersion = llvm::TargetMachine::parseBinutilsVersion(BinutilsVersion);
        Options.DisableIntegratedAS = false;
        // TODO: investigate
        //Options.DisableIntegratedAS = NoIntegratedAssembler;
        Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
        Options.MCOptions.AsmVerbose = AsmVerbose;
        Options.MCOptions.PreserveAsmComments = PreserveComments;
        Options.MCOptions.IASSearchPaths = IncludeDirs;
        Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
        if (DwarfDirectory.getPosition()) 
        {
            Options.MCOptions.MCUseDwarfDirectory =
            DwarfDirectory ? llvm::MCTargetOptions::EnableDwarfDirectory
                           : llvm::MCTargetOptions::DisableDwarfDirectory;
        } else {
            // -dwarf-directory is not set explicitly. Some assemblers
            // (e.g. GNU as or ptxas) do not support `.file directory'
            // syntax prior to DWARFv5. Let the target decide the default
            // value.
            Options.MCOptions.MCUseDwarfDirectory = llvm::MCTargetOptions::DefaultDwarfDirectory;
        }
    };

    std::optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();
    std::optional<llvm::CodeModel::Model> CM = llvm::codegen::getExplicitCodeModel();

    llvm::Triple TheTriple;
    const llvm::Target *TheTarget = nullptr;
    
    // If we are supposed to override the target triple, do so now.
    std::string llvmModuleTargetTriple = llvmModule->getTargetTriple();
    if (!TargetTriple.empty())
    {
        llvmModuleTargetTriple = llvm::Triple::normalize(TargetTriple);
    }
    
    TheTriple = llvm::Triple(llvmModuleTargetTriple);
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

    // TODO: research
    if (RM && *RM == llvm::Reloc::PIC_)
    {
        llvmModule->setPICLevel(llvm::PICLevel::Level::BigPIC);
        //llvmModule->setPIELevel(llvm::PIELevel::Level::Large);
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

    assert(llvmModule && "Should have exited if we didn't have a module!");
    // TODO: seems it should be used in Target
    if (llvm::codegen::getFloatABIForCalls() != llvm::FloatABI::Default)
    {
        Options.FloatABIType = llvm::codegen::getFloatABIForCalls();
    }

    // setting values to module
    llvmModule->setDataLayout(Target->createDataLayout());
    llvmModule->setTargetTriple(TheTriple.getTriple());

    // Override function attributes based on CPUStr, FeaturesStr, and command line flags.
    llvm::codegen::setFunctionAttributes(CPUStr, FeaturesStr, *llvmModule);

    return 0;
}

int dumpObjOrAssembly(int argc, char **argv, enum Action emitAction, std::string outputFile, mlir::ModuleOp module, CompileOptions &compileOptions)
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
    //mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());    

    llvm::TargetOptions Options;
    std::unique_ptr<llvm::TargetMachine> Target;
    auto retCode = setupTargetTriple(llvmModule.get(), Target, Options);
    if (retCode != 0)
    {
        return retCode;
    }

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel, compileOptions);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    //
    // generate Obj
    //
    llvm::LLVMContext Context;
    Context.setDiscardValueNames(DiscardValueNames);

    // Set a diagnostic handler that doesn't exit on the first error
    bool HasError = false;
    Context.setDiagnosticHandler(std::make_unique<LLCDiagnosticHandler>(&HasError));

    auto FDOut = getOutputStream(emitAction, outputFile);
    if (!FDOut)
    {
        return -1;
    }

    // Ensure the filename is passed down to CodeViewDebug.
    Target->Options.ObjectFilenameForDebug = FDOut->outputFilename();

    std::unique_ptr<llvm::ToolOutputFile> DwoOut;
    if (!SplitDwarfOutputFile.empty()) {
        std::error_code EC;
        DwoOut = std::make_unique<llvm::ToolOutputFile>(SplitDwarfOutputFile, EC, llvm::sys::fs::OF_None);
        if (EC)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << EC.message() << SplitDwarfOutputFile << "\n";
            return -1;
        }
    }

    // Build up all of the passes that we want to do to the module.
    llvm::legacy::PassManager PM;

    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(llvmModule->getTargetTriple()));

    PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

    if (!NoVerify && llvm::verifyModule(*llvmModule.get(), &llvm::errs()))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "input module cannot be verified\n";
        return -1;        
    }

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
                        PM, *OS, DwoOut ? &DwoOut->os() : nullptr,
                        fileFormat /*llvm::codegen::getFileType()*/, true, MMIWP)) 
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "target does not support generation of this file type\n";
        }

        const_cast<llvm::TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())->Initialize(MMIWP->getMMI().getContext(), *Target);

        // Before executing passes, print the final values of the LLVM options.
        //llvm::cl::PrintOptionValues();

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
    if (DwoOut)
    {
        DwoOut->keep();
    }

    return 0;
}

int dumpObjOrAssembly(int argc, char **argv, mlir::ModuleOp module, CompileOptions &compileOptions)
{
    std::string fileOutput = outputFilename.empty() ? getDefaultOutputFileName(emitAction) : outputFilename;
    return dumpObjOrAssembly(argc, argv, emitAction, fileOutput, module, compileOptions);
}
