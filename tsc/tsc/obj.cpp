#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

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
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;

std::unique_ptr<llvm::ToolOutputFile> getOutputStream();
int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int);

static llvm::codegen::RegisterCodeGenFlags CGF;

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
