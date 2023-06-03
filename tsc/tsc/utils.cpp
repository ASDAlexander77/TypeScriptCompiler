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

std::unique_ptr<llvm::ToolOutputFile> getOutputStream()
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
