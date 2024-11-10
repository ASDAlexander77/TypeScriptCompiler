#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/Host.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"

#define DEBUG_TYPE "tsc"

namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<std::string> outputFilename;
extern cl::opt<enum Action> emitAction;

// obj
extern cl::opt<std::string> TargetTriple;

std::string getDefaultExt(enum Action emitAction)
{
    std::string ext;

    switch (emitAction)
    {
        case ActionNotSet:
        case RunJIT:
            break;
        case DumpAST:
            ext = "txt";
            break;
        case DumpMLIR:
        case DumpMLIRAffine:
        case DumpMLIRLLVM:
            ext = "mlir";
            break;
        case DumpLLVMIR:
            ext = "ll";
            break;
        case DumpByteCode:
            ext = "bc";
            break;
        case DumpObj:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                ext = (TheTriple.getOS() == llvm::Triple::Win32) ? "obj" : "o";
            }

            break;
        case DumpAssembly:
            ext = "s";
            break;
        case BuildDll:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                ext = (TheTriple.getOS() == llvm::Triple::Win32) ? "dll" : "so";
            }

            break;
        case BuildExe:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                ext = 
                    (TheTriple.getOS() == llvm::Triple::Win32) 
                        ? "exe" 
                        : (TheTriple.getArch() == llvm::Triple::wasm32 || TheTriple.getArch() == llvm::Triple::wasm64) 
                            ? "wasm" 
                            : "";
            }

            break;
    }

    return ext;    
}

std::string getDefaultOutputFileName(enum Action emitAction)
{
    if (inputFilename == "-")
    {
        return "-";
    }

    std::string fileNameResult;

    // If InputFilename ends in .bc or .ll, remove it.
    llvm::StringRef IFN = inputFilename;
    if (IFN.ends_with(".ts"))
        fileNameResult = std::string(IFN.drop_back(3));
    else if (IFN.ends_with(".mlir"))
        fileNameResult = std::string(IFN.drop_back(5));
    else
        fileNameResult = std::string(IFN);

    switch (emitAction)
    {
        case ActionNotSet:
            fileNameResult = "-";
            break;
        case DumpAST:
            fileNameResult += ".txt";
            break;
        case DumpMLIR:
        case DumpMLIRAffine:
        case DumpMLIRLLVM:
            fileNameResult += ".mlir";
            break;
        case DumpLLVMIR:
            fileNameResult += ".ll";
            break;
        case DumpByteCode:
            fileNameResult += ".bc";
            break;
        case DumpObj:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                fileNameResult += (TheTriple.getOS() == llvm::Triple::Win32) ? ".obj" : ".o";
            }

            break;
        case DumpAssembly:
            fileNameResult += ".s";
            break;
        case BuildDll:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                fileNameResult += (TheTriple.getOS() == llvm::Triple::Win32) ? ".dll" : ".so";
                if ((TheTriple.getOS() != llvm::Triple::Win32))
                {
                    fileNameResult.insert(0, "lib");
                }
            }

            break;
        case BuildExe:
            {
                llvm::Triple TheTriple;
                std::string targetTriple = llvm::sys::getDefaultTargetTriple();
                if (!TargetTriple.empty())
                {
                    targetTriple = llvm::Triple::normalize(TargetTriple);
                }
                
                TheTriple = llvm::Triple(targetTriple);

                fileNameResult += 
                    (TheTriple.getOS() == llvm::Triple::Win32) 
                        ? ".exe" 
                        : (TheTriple.getArch() == llvm::Triple::wasm32 || TheTriple.getArch() == llvm::Triple::wasm64) 
                            ? ".wasm" 
                            : "";
            }

            break;
        case RunJIT:
            fileNameResult = "-";
            break;
    }

    return fileNameResult;
}

std::unique_ptr<llvm::ToolOutputFile> getOutputStream(enum Action emitAction, std::string outputFilename)
{
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
