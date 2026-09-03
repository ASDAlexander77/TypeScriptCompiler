#ifndef TYPESCRIPT_COMPILER_DEFINES_H_
#define TYPESCRIPT_COMPILER_DEFINES_H_

enum Action
{
    ActionNotSet,
    DumpAST,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    DumpByteCode,
    DumpObj,
    DumpAssembly,
    BuildExe,
    BuildDll,
    RunJIT
};

enum Exports
{
    ExportsNotSet,
    ExportAll,
    IgnoreAll
};

// How compiled code reclaims heap memory. There have always been three of these - `-nogc`
// meant "leak everything", not "collect differently" - but they were spelled as one boolean.
// See docs/reference-counting-evaluation.md.
enum MemoryModel
{
    // Boehm-Demers-Weiser collector. The default, and the only model that reclaims today.
    MemoryModelGC,
    // Reference counting. In development: counts are maintained and the release machinery is
    // generated, but nothing inserts retains or releases yet, so the collector still runs and
    // is still what actually frees. See section 9.6.
    MemoryModelRC,
    // No reclamation at all.
    MemoryModelNone
};

// The spelling used both by the `-mm=` flag and by the shared-library marker symbol, so the
// two can never disagree about what a model is called.
inline const char *memoryModelName(enum MemoryModel model)
{
    switch (model)
    {
    case MemoryModelRC:
        return "rc";
    case MemoryModelNone:
        return "none";
    default:
        return "gc";
    }
}

#endif // TYPESCRIPT_COMPILER_DEFINES_H_