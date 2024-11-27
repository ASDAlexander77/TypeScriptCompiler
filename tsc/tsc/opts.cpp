#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/Host.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/DataStructs.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> disableGC;
extern cl::opt<bool> disableWarnings;
extern cl::opt<bool> generateDebugInfo;
extern cl::opt<bool> lldbDebugInfo;
extern cl::opt<std::string> TargetTriple;
extern cl::opt<enum Exports> exportAction;
extern cl::opt<bool> enableBuiltins;
extern cl::opt<bool> noDefaultLib;
extern cl::opt<std::string> outputFilename;
extern cl::opt<bool> appendGCtorsToMethod;

// obj
extern cl::opt<std::string> TargetTriple;

CompileOptions prepareOptions()
{
    auto moduleTargetTriple = TargetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : llvm::Triple::normalize(TargetTriple);

    auto TheTriple = llvm::Triple(moduleTargetTriple);

    CompileOptions compileOptions;
    compileOptions.isJit = emitAction.getValue() == Action::RunJIT;
    compileOptions.disableGC = disableGC.getValue();
    compileOptions.enableBuiltins = enableBuiltins.getValue();
    compileOptions.noDefaultLib = noDefaultLib.getValue();
    compileOptions.disableWarnings = disableWarnings.getValue();
    compileOptions.exportOpt = exportAction.getValue();
    compileOptions.generateDebugInfo = generateDebugInfo.getValue();
    compileOptions.lldbDebugInfo = lldbDebugInfo.getValue();
    compileOptions.moduleTargetTriple = moduleTargetTriple;
    compileOptions.isWindows = TheTriple.isKnownWindowsMSVCEnvironment();
    compileOptions.isWasm = TheTriple.getArch() == llvm::Triple::wasm64 || TheTriple.getArch() == llvm::Triple::wasm32;
    compileOptions.sizeBits = 32;
    compileOptions.isExecutable = emitAction == Action::BuildExe;
    compileOptions.isDLL = emitAction == Action::BuildDll;
    compileOptions.appendGCtorsToMethod = appendGCtorsToMethod.getValue();
    if (
        TheTriple.getArch() == llvm::Triple::UnknownArch
        || TheTriple.getArch() == llvm::Triple::aarch64        // AArch64 (little endian): aarch64
        || TheTriple.getArch() == llvm::Triple::aarch64_be     // AArch64 (big endian): aarch64_be
        || TheTriple.getArch() == llvm::Triple::aarch64_32     // AArch64 (little endian) ILP32: aarch64_32
        || TheTriple.getArch() == llvm::Triple::bpfel          // eBPF or extended BPF or 64-bit BPF (little endian)
        || TheTriple.getArch() == llvm::Triple::bpfeb          // eBPF or extended BPF or 64-bit BPF (big endian)
        || TheTriple.getArch() == llvm::Triple::loongarch64    // LoongArch (64-bit): loongarch64
        || TheTriple.getArch() == llvm::Triple::mips64         // MIPS64: mips64, mips64r6, mipsn32, mipsn32r6
        || TheTriple.getArch() == llvm::Triple::mips64el       // MIPS64EL: mips64el, mips64r6el, mipsn32el, mipsn32r6el
        || TheTriple.getArch() == llvm::Triple::ppc64          // PPC64: powerpc64, ppu
        || TheTriple.getArch() == llvm::Triple::ppc64le        // PPC64LE: powerpc64le
        || TheTriple.getArch() == llvm::Triple::riscv64        // RISC-V (64-bit): riscv64
        || TheTriple.getArch() == llvm::Triple::x86_64         // X86-64: amd64, x86_64
        || TheTriple.getArch() == llvm::Triple::nvptx64        // NVPTX: 64-bit
        || TheTriple.getArch() == llvm::Triple::le64           // le64: generic little-endian 64-bit CPU (PNaCl)
        || TheTriple.getArch() == llvm::Triple::amdil64        // AMDIL with 64-bit pointers
        || TheTriple.getArch() == llvm::Triple::hsail64        // AMD HSAIL with 64-bit pointers
        || TheTriple.getArch() == llvm::Triple::spir64         // SPIR: standard portable IR for OpenCL 64-bit version
        || TheTriple.getArch() == llvm::Triple::spirv64        // SPIR-V with 64-bit pointers
        || TheTriple.getArch() == llvm::Triple::wasm64         // WebAssembly with 64-bit pointers
        || TheTriple.getArch() == llvm::Triple::renderscript64 // 64-bit RenderScript
    ) {        
        compileOptions.sizeBits = 64;
    }

    if (!outputFilename.empty())
    {
        llvm::SmallString<256> outputPath(outputFilename);
        llvm::sys::path::remove_filename(outputPath);    
        compileOptions.outputFolder = outputPath.str().str();
    }

    return compileOptions;
}
