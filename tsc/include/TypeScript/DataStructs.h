#ifndef TYPESCRIPT_DATASTRUCT_H_
#define TYPESCRIPT_DATASTRUCT_H_

#include "TypeScript/TypeScriptCompiler/Defines.h"

#include <string>

struct CompileOptions
{
    bool isJit;
    bool disableGC;
    bool enableBuiltins;
    bool noDefaultLib;
    std::string defaultDeclarationTSFile;
    bool disableWarnings;
    bool generateDebugInfo;
    bool lldbDebugInfo;
    std::string moduleTargetTriple;
    int sizeBits;
    bool isWasm;
    bool isWindows;
    enum Exports exportOpt;
    std::string outputFolder;
};

#endif // TYPESCRIPT_DATASTRUCT_H_