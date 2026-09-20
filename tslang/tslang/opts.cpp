#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/Host.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/TargetInfo.h"

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<enum Action> emitAction;
extern cl::opt<enum MemoryModel> memoryModelOpt;
extern cl::opt<bool> disableWarnings;
extern cl::opt<bool> generateDebugInfo;
extern cl::opt<bool> lldbDebugInfo;
extern cl::opt<std::string> TargetTriple;
extern cl::list<std::string> exportFilters;
extern cl::opt<bool> enableBuiltins;
extern cl::opt<bool> noDefaultLib;
extern cl::opt<std::string> outputFilename;
extern cl::opt<bool> appendGCtorsToMethod;
extern cl::opt<bool> entryPoint;
extern cl::opt<bool> strictNullChecks;
extern cl::opt<bool> embedExportDeclarationsAction;
extern cl::opt<bool> enableFastMath;

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
    compileOptions.memoryModel = memoryModelOpt.getValue();
    compileOptions.enableBuiltins = enableBuiltins.getValue();
    compileOptions.noDefaultLib = noDefaultLib.getValue();
    compileOptions.disableWarnings = disableWarnings.getValue();
    compileOptions.exportFilters.assign(exportFilters.begin(), exportFilters.end());
    compileOptions.embedExportDeclarations = embedExportDeclarationsAction.getValue();
    compileOptions.generateDebugInfo = generateDebugInfo.getValue();
    compileOptions.lldbDebugInfo = lldbDebugInfo.getValue();
    compileOptions.moduleTargetTriple = moduleTargetTriple;
    compileOptions.isWindows = TheTriple.isKnownWindowsMSVCEnvironment();
    compileOptions.isWasm = TheTriple.getArch() == llvm::Triple::wasm64 || TheTriple.getArch() == llvm::Triple::wasm32;
    compileOptions.targetInfo = TargetInfo::fromTriple(
        TheTriple, llvm::Triple(llvm::sys::getDefaultTargetTriple()));
    compileOptions.isExecutable = emitAction == Action::BuildExe;
    compileOptions.isDLL = emitAction == Action::BuildDll;
    // A DLL never gets one, whatever was asked for: its root initialization runs from the
    // global constructors and there is no program here to enter.
    compileOptions.generateEntryPoint = !compileOptions.isDLL &&
        (compileOptions.isJit || compileOptions.isExecutable || entryPoint.getValue());
    compileOptions.appendGCtorsToMethod = appendGCtorsToMethod.getValue();
    compileOptions.strictNullChecks = strictNullChecks.getValue();
    compileOptions.enableFastMath = enableFastMath.getValue();

    if (!outputFilename.empty())
    {
        llvm::SmallString<256> outputPath(outputFilename);
        llvm::sys::path::remove_filename(outputPath);    
        compileOptions.outputFolder = outputPath.str().str();
    }

    return compileOptions;
}
