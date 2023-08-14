#ifndef DATASTRUCT_H_
#define DATASTRUCT_H_

#include <string>

struct CompileOptions
{
    bool isJit;
    bool disableGC;
    bool disableWarnings;
    bool generateDebugInfo;
    bool lldbDebugInfo;
    std::string moduleTargetTriple;
    int sizeBits;
    bool isWasm;
};

#endif // DATASTRUCT_H_