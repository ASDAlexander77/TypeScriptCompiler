#include "TypeScript/MLIRGen.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/TargetSelect.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<bool> disableGC;
extern cl::opt<bool> disableWarnings;
extern cl::opt<bool> generateDebugInfo;
extern cl::opt<bool> lldbDebugInfo;
extern cl::opt<std::string> TargetTriple;

int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, mlir::OwningOpRef<mlir::ModuleOp> &module)
{
    auto fileName = llvm::StringRef(inputFilename);

    // Handle '.ts' input to the compiler.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    auto moduleTargetTriple = TargetTriple.empty() 
        ? llvm::sys::getDefaultTargetTriple() 
        : llvm::Triple::normalize(TargetTriple);

    auto TheTriple = llvm::Triple(moduleTargetTriple);

    CompileOptions compileOptions;
    compileOptions.disableGC = disableGC;
    compileOptions.disableWarnings = disableWarnings;
    compileOptions.generateDebugInfo = generateDebugInfo;
    compileOptions.lldbDebugInfo = lldbDebugInfo;
    compileOptions.moduleTargetTriple = moduleTargetTriple;
    compileOptions.sizeBits = 32;
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
    
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    module = mlirGenFromSource(context, fileName, sourceMgr, compileOptions);
    return !module ? 1 : 0;
}
