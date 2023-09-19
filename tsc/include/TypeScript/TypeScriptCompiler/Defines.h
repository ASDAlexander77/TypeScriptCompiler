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

#endif // TYPESCRIPT_COMPILER_DEFINES_H_