#ifndef DATASTRUCT_H_
#define DATASTRUCT_H_

#include <string>

struct CompileOptions
{
    bool disableGC;
    bool generateDebugInfo;
    bool lldbDebugInfo;
    std::string moduleTargetTriple;
};

#endif // DATASTRUCT_H_