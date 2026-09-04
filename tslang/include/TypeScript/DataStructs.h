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

    // Whether the Boehm runtime has to be present. Only `gc` needs it: it is the model whose
    // reclamation *is* the collector. `rc` frees through the reference counts it maintains and
    // `none` frees nothing, so both allocate straight from `malloc` and neither links libgc.
    //
    // This was true for `rc` too while the retain/release insertion points were being built
    // (§9.6 through §9.27). Boehm collecting behind them made a missing release invisible, which
    // was the point at the time - but it also meant no memory measurement taken under `rc` said
    // anything about reference counting, since the collector was doing the reclaiming either
    // way. See docs/reference-counting-evaluation.md §9.28.
    bool needsGCRuntime() const
    {
        return memoryModel == MemoryModelGC;
    }

    // Whether allocations maintain a reference count in the block header.
    bool isRefCounted() const
    {
        return memoryModel == MemoryModelRC;
    }
};

#endif // TYPESCRIPT_DATASTRUCT_H_