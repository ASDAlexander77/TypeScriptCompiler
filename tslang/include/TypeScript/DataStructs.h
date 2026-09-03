#ifndef TYPESCRIPT_DATASTRUCT_H_
#define TYPESCRIPT_DATASTRUCT_H_

#include "TypeScript/TypeScriptCompiler/Defines.h"

#include <string>

struct CompileOptions
{
    bool isJit;
    enum MemoryModel memoryModel;
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
    bool isExecutable;
    bool isDLL;
    enum Exports exportOpt;
    bool embedExportDeclarations;
    std::string outputFolder;
    bool appendGCtorsToMethod;
    bool strictNullChecks;
    bool enableFastMath;

    // Whether the Boehm runtime has to be present: it is what reclaims under both `gc` and,
    // for now, `rc`.
    bool needsGCRuntime() const
    {
        return memoryModel != MemoryModelNone;
    }

    // Whether allocations maintain a reference count in the block header.
    bool isRefCounted() const
    {
        return memoryModel == MemoryModelRC;
    }
};

#endif // TYPESCRIPT_DATASTRUCT_H_