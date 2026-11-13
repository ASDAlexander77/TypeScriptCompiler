#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

// for dump obj
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
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

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::opt<std::string> outputFilename;

std::string getDefaultOutputFileName(enum Action);
std::unique_ptr<llvm::ToolOutputFile> getOutputStream(enum Action, std::string);
int registerMLIRDialects(mlir::ModuleOp);
namespace llvm { class TargetMachine; }
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int, CompileOptions &, llvm::TargetMachine *);

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
    std::string llvmModuleTargetTriple = llvmModule->getTargetTriple().str();
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
        llvm::WithColor::error(llvm::errs(), "tslang") << Error;
        return -1;
    }

      // On AIX, setting the relocation model to anything other than PIC is
      // considered a user error.
    if (TheTriple.isOSAIX() && RM && *RM != llvm::Reloc::PIC_)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "invalid relocation model, AIX only supports PIC";
        return -1;
    }

    // Android executables must be PIE, and a shared library is PIC anyway: with LLVM's static
    // default the objects would not link (buildExe makes the same choice for the link).
    if (!RM && TheTriple.isAndroid())
    {
        RM = llvm::Reloc::PIC_;
    }

    // TODO: research
    if (RM && *RM == llvm::Reloc::PIC_)
    {
        llvmModule->setPICLevel(llvm::PICLevel::Level::BigPIC);
        //llvmModule->setPIELevel(llvm::PIELevel::Level::Large);
    }

    auto CPUStr = llvm::codegen::getCPUStr(),
         FeaturesStr = llvm::codegen::getFeaturesStr();

    // wasm (not Emscripten): try/catch is wasm exception handling - see WasmExceptionPass. The
    // backend takes it from -wasm-enable-eh (which picks the wasm exception model), and the
    // instructions from the exception-handling feature.
    if (TheTriple.isWasm() && !TheTriple.isOSEmscripten())
    {
        auto &registeredOptions = llvm::cl::getRegisteredOptions();
        if (auto it = registeredOptions.find("wasm-enable-eh"); it != registeredOptions.end())
        {
            static_cast<llvm::cl::opt<bool> *>(it->second)->setValue(true);
        }

        // Threads share the module's memory: a program that starts one is linked with a shared memory
        // (see exe.cpp), which every object in it - the default library's too - must allow, by having
        // real atomics and passive data segments. Atomics work on an unshared memory as well.
        FeaturesStr += FeaturesStr.empty() ? "+exception-handling" : ",+exception-handling";
        FeaturesStr += ",+atomics,+bulk-memory";
    }

    llvm::CodeGenOptLevel OLvl;
    // TODO: finish it, optimization
    // optimizing for size generates code as -O2 does, whatever the speed level: clang's -Os and -Oz
    auto codeGenLevel = !enableOpt ? '0' : (sizeLevel > 0 && optLevel > 0) ? '2' : mapToLevel(optLevel);
    if (auto Level = llvm::CodeGenOpt::parseLevel(codeGenLevel)) 
    {
        OLvl = *Level;
    } 
    else 
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "invalid optimization level.\n";
        return 1;
    }

    InitializeOptions(TheTriple);

    // Every function and global in a section of its own (a COMDAT on Windows), as clang does with
    // -ffunction-sections -fdata-sections: the linker can then drop what a program never uses. Off,
    // a module is one section, and a program that calls one default-library function linked all of
    // lib.ts and, through its references, the RegExp, Date and HTTP wrappers too. --function-sections
    // / --data-sections given on the command line still win.
    auto &registeredOptions = llvm::cl::getRegisteredOptions();
    auto notGiven = [&](llvm::StringRef name) {
        auto it = registeredOptions.find(name);
        return it == registeredOptions.end() || it->second->getNumOccurrences() == 0;
    };
    if (notGiven("function-sections"))
    {
        Options.FunctionSections = true;
    }

    if (notGiven("data-sections"))
    {
        Options.DataSections = true;
    }

    // before the target machine is made: its MCAsmInfo takes the exception model from here, and
    // with none the code generator lowers every invoke to a call (see wasm-enable-eh above)
    if (TheTriple.isWasm() && !TheTriple.isOSEmscripten())
    {
        Options.ExceptionModel = llvm::ExceptionHandling::Wasm;
    }

    Target =std::unique_ptr<llvm::TargetMachine>(TheTarget->createTargetMachine(
          TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");

    assert(llvmModule && "Should have exited if we didn't have a module!");
    // TODO: seems it should be used in Target
    if (llvm::codegen::getFloatABIForCalls() != llvm::FloatABI::Default)
    {
        Options.FloatABIType = llvm::codegen::getFloatABIForCalls();
    }

    // setting values to module
    llvmModule->setDataLayout(Target->createDataLayout());
    llvmModule->setTargetTriple(TheTriple);

    // Override function attributes based on CPUStr, FeaturesStr, and command line flags.
    llvm::codegen::setFunctionAttributes(CPUStr, FeaturesStr, *llvmModule);

    return 0;
}

// wasm: copies and fills of plain loads and stores, not memory.copy and memory.fill.
//
// V8 (Node 26, wasm32 and wasm64 alike) checks memory.copy and memory.fill against the size of a
// shared memory as this thread last saw it: right after another thread grew the memory, they trap
// ("memory access out of bounds") on a block that thread has just allocated - a queue's array after
// the other thread's push - while plain loads and stores see the new size at once. LLVM emits them
// for every copy and fill it doesn't expand into loads and stores once bulk-memory is on, which
// passive data segments - a shared memory - need, and nothing turns them off on their own. So each
// llvm.memcpy, memmove and memset that would become one is a call to one of these internal functions
// instead. A copy of a small constant size is left alone: the backend expands it into loads and
// stores. (A byval argument's copy is the backend's own, but it is one thread's stack to its stack.)
static llvm::Function *getWasmMemoryFunction(llvm::Module &module, bool fill)
{
    auto name = fill ? "__tslang_wasm_memset" : "__tslang_wasm_memmove";
    if (auto *existing = module.getFunction(name))
    {
        return existing;
    }

    auto &context = module.getContext();
    auto *ptrTy = llvm::PointerType::get(context, 0);
    auto *sizeTy = module.getDataLayout().getIntPtrType(context);
    auto *i8Ty = llvm::Type::getInt8Ty(context);
    auto *i64Ty = llvm::Type::getInt64Ty(context);
    auto *fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                         {ptrTy, fill ? (llvm::Type *)i8Ty : ptrTy, sizeTy}, false);
    auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, module);
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr("no-builtins");

    auto *dst = fn->getArg(0);
    auto *src = fn->getArg(1);
    auto *count = fn->getArg(2);
    auto *eight = llvm::ConstantInt::get(sizeTy, 8);
    auto *one = llvm::ConstantInt::get(sizeTy, 1);
    auto *zero = llvm::ConstantInt::get(sizeTy, 0);

    auto *entry = llvm::BasicBlock::Create(context, "entry", fn);
    auto *done = llvm::BasicBlock::Create(context, "done", fn);
    llvm::IRBuilder<> builder(done);
    builder.CreateRetVoid();

    // one unit at `at`: 8 bytes or 1, unaligned (wasm loads and stores take any address)
    auto unit = [&](llvm::IRBuilder<> &b, llvm::Value *at, llvm::Value *word, bool wide) {
        auto *type = wide ? i64Ty : i8Ty;
        auto *to = b.CreateGEP(i8Ty, dst, at);
        auto *value = fill ? (wide ? word : (llvm::Value *)src)
                           : (llvm::Value *)b.CreateAlignedLoad(type, b.CreateGEP(i8Ty, src, at), llvm::MaybeAlign(1));
        b.CreateAlignedStore(value, to, llvm::MaybeAlign(1));
    };

    // up: words then bytes, from the start; down (an overlapping copy to a higher address): from the end
    auto loops = [&](llvm::BasicBlock *from, llvm::Value *word, bool up) {
        auto *words = llvm::BasicBlock::Create(context, up ? "up.words" : "down.words", fn);
        auto *bytesCheck = llvm::BasicBlock::Create(context, up ? "up.bytes.check" : "down.bytes.check", fn);
        auto *bytes = llvm::BasicBlock::Create(context, up ? "up.bytes" : "down.bytes", fn);

        llvm::IRBuilder<> b(from);
        auto *start = up ? (llvm::Value *)zero : (llvm::Value *)count;
        auto *wordsCheck = llvm::BasicBlock::Create(context, up ? "up.words.check" : "down.words.check", fn, words);
        b.CreateBr(wordsCheck);

        // words: up while at + 8 <= count, down while at >= 8
        b.SetInsertPoint(wordsCheck);
        auto *at = b.CreatePHI(sizeTy, 2);
        at->addIncoming(start, from);
        auto *more = up ? b.CreateICmpULE(b.CreateAdd(at, eight), count) : b.CreateICmpUGE(at, eight);
        b.CreateCondBr(more, words, bytesCheck);

        b.SetInsertPoint(words);
        auto *wordAt = up ? (llvm::Value *)at : b.CreateSub(at, eight);
        unit(b, wordAt, word, true);
        auto *nextWord = up ? b.CreateAdd(at, eight) : wordAt;
        at->addIncoming(nextWord, words);
        b.CreateBr(wordsCheck);

        // bytes: up while at < count, down while at > 0
        b.SetInsertPoint(bytesCheck);
        auto *byteAt = b.CreatePHI(sizeTy, 2);
        byteAt->addIncoming(at, wordsCheck);
        auto *moreBytes = up ? b.CreateICmpULT(byteAt, count) : b.CreateICmpUGT(byteAt, zero);
        b.CreateCondBr(moreBytes, bytes, done);

        b.SetInsertPoint(bytes);
        auto *thisByte = up ? (llvm::Value *)byteAt : b.CreateSub(byteAt, one);
        unit(b, thisByte, word, false);
        byteAt->addIncoming(up ? b.CreateAdd(byteAt, one) : thisByte, bytes);
        b.CreateBr(bytesCheck);
    };

    builder.SetInsertPoint(entry);
    if (fill)
    {
        // the byte in each of the word's 8
        auto *word = builder.CreateMul(builder.CreateZExt(src, i64Ty), llvm::ConstantInt::get(i64Ty, 0x0101010101010101ULL));
        auto *body = llvm::BasicBlock::Create(context, "fill", fn, done);
        builder.CreateBr(body);
        loops(body, word, true);
        return fn;
    }

    // to a lower address (or no overlap) up from the start, to a higher one down from the end
    auto *up = llvm::BasicBlock::Create(context, "up", fn, done);
    auto *down = llvm::BasicBlock::Create(context, "down", fn, done);
    auto *dstAddress = builder.CreatePtrToInt(dst, sizeTy);
    auto *srcAddress = builder.CreatePtrToInt(src, sizeTy);
    builder.CreateCondBr(builder.CreateICmpULE(dstAddress, srcAddress), up, down);
    loops(up, nullptr, true);
    loops(down, nullptr, false);
    return fn;
}

static void replaceWasmBulkMemory(llvm::Module &module)
{
    // the backend expands a constant copy or fill this small (at most 4 stores of 8 bytes, as at -Os)
    // into loads and stores
    const uint64_t expandedSize = 32;

    llvm::SmallVector<llvm::MemIntrinsic *> calls;
    for (auto &function : module)
    {
        for (auto &instruction : llvm::instructions(function))
        {
            if (auto *call = llvm::dyn_cast<llvm::MemIntrinsic>(&instruction))
            {
                auto *size = llvm::dyn_cast<llvm::ConstantInt>(call->getLength());
                if (!size || size->getZExtValue() > expandedSize)
                {
                    calls.push_back(call);
                }
            }
        }
    }

    for (auto *call : calls)
    {
        auto fill = llvm::isa<llvm::MemSetInst>(call);
        auto *fn = getWasmMemoryFunction(module, fill);
        llvm::IRBuilder<> builder(call);
        auto *length = builder.CreateZExtOrTrunc(call->getLength(), fn->getArg(2)->getType());
        auto *second = fill ? llvm::cast<llvm::MemSetInst>(call)->getValue()
                            : llvm::cast<llvm::MemTransferInst>(call)->getSource();
        auto *replacement = builder.CreateCall(fn, {call->getDest(), second, length});
        replacement->setDebugLoc(call->getDebugLoc());
        call->eraseFromParent();
    }
}

int dumpObjOrAssembly(int argc, char **argv, enum Action emitAction, std::string outputFile, mlir::ModuleOp module, CompileOptions &compileOptions)
{
    registerMLIRDialects(module);

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "Failed to emit LLVM IR\n";
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

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel, compileOptions, Target.get());
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    // a wasm program that starts a thread (Thread.spawn, new Worker) is linked with a shared memory
    if (compileOptions.wasmExceptions && llvmModule->getFunction("tslang_thread_spawn"))
    {
        compileOptions.wasmThreads = true;
    }

    // after the optimizations, which would turn the loops back into llvm.memcpy
    if (auto triple = llvm::Triple(llvmModule->getTargetTriple()); triple.isWasm() && !triple.isOSEmscripten())
    {
        replaceWasmBulkMemory(*llvmModule);
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
            llvm::WithColor::error(llvm::errs(), "tslang") << EC.message() << SplitDwarfOutputFile << "\n";
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
        llvm::WithColor::error(llvm::errs(), "tslang") << "input module cannot be verified\n";
        return -1;        
    }

    auto fileFormat = emitAction == DumpObj ? llvm::CodeGenFileType::ObjectFile : emitAction == DumpAssembly ? llvm::CodeGenFileType::AssemblyFile : llvm::CodeGenFileType::Null;
    if (llvm::mc::getExplicitRelaxAll() && /*llvm::codegen::getFileType()*/ fileFormat != llvm::CodeGenFileType::ObjectFile)
    {
        llvm::WithColor::warning(llvm::errs(), "tslang") << ": warning: ignoring -mc-relax-all because filetype != obj";
    }

    {
        llvm::raw_pwrite_stream *OS = &FDOut->os();

        // Manually do the buffering rather than using buffer_ostream,
        // so we can memcmp the contents in CompileTwice mode
        llvm::SmallVector<char, 0> Buffer;
        std::unique_ptr<llvm::raw_svector_ostream> BOS;
        if ((/*llvm::codegen::getFileType()*/ fileFormat != llvm::CodeGenFileType::AssemblyFile &&
            !FDOut->os().supportsSeeking())) 
        {
            BOS = std::make_unique<llvm::raw_svector_ostream>(Buffer);
            OS = BOS.get();
        }

        auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(Target.get());

        if (Target->addPassesToEmitFile(
                        PM, *OS, DwoOut ? &DwoOut->os() : nullptr,
                        fileFormat /*llvm::codegen::getFileType()*/, true, MMIWP))
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "target does not support generation of this file type\n";
        }

        const_cast<llvm::TargetLoweringObjectFile *>(Target->getObjFileLowering())->Initialize(MMIWP->getMMI().getContext(), *Target);

        // Before executing passes, print the final values of the LLVM options.
        //llvm::cl::PrintOptionValues();

        PM.run(*llvmModule.get());

        auto HasError = ((const LLCDiagnosticHandler *)(Context.getDiagHandlerPtr()))->HasError;
        if (*HasError)
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "diagnostic error(s)\n";
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
