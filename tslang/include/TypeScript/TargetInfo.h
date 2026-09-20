#ifndef TYPESCRIPT_TARGETINFO_H_
#define TYPESCRIPT_TARGETINFO_H_

#include "llvm/TargetParser/Triple.h"

// Everything the compiler needs to know about the target beyond the triple string itself,
// derived once in prepareOptions() and carried on CompileOptions.
//
// Predicates are named for the decision they answer, never for the architecture that currently
// needs them - `usesImageBaseRelativeEH`, not `isX64`. A new target then answers the questions,
// instead of every call site growing another arch comparison. This replaces a hand-maintained
// list of 64-bit arches in opts.cpp that had already rotted.
struct TargetInfo
{
    // Width of a pointer, and of the integer type used for sizes and indices.
    unsigned pointerBits = 64;

    // MSVC C++ EH stores the cross-references inside ThrowInfo and CatchableType as
    // image-base-relative RVAs on 64-bit Windows, and as absolute pointers on 32-bit x86. Both
    // are 4 bytes, so the struct layouts coincide and only the stored value differs: x86 must
    // not subtract the image base. See Phase 3.
    bool usesImageBaseRelativeEH = true;

    // _CxxThrowException is __stdcall on 32-bit x86, so the emitted symbol carries the @8
    // suffix that the linker asks for.
    bool stdcallDecoratesCxxThrow = false;

    // Whether tslang.exe - an x64 process - can execute this target's code in-process.
    bool supportsInProcessJit = true;

    static TargetInfo fromTriple(const llvm::Triple &target, const llvm::Triple &host)
    {
        TargetInfo info;

        info.pointerBits = target.getArchPointerBitWidth();
        if (info.pointerBits == 0)
        {
            // An unknown arch has no width of its own. Zero would make every size computation
            // degenerate, so take the host's - the same assumption the compiler already makes
            // when no triple is given at all.
            info.pointerBits = host.getArchPointerBitWidth();
        }

        const bool msvc = target.isKnownWindowsMSVCEnvironment();
        const bool x86_32 = target.getArch() == llvm::Triple::x86;

        info.usesImageBaseRelativeEH = msvc && !x86_32;
        info.stdcallDecoratesCxxThrow = msvc && x86_32;
        info.supportsInProcessJit =
            target.getArch() == host.getArch() && target.getOS() == host.getOS();

        return info;
    }
};

#endif // TYPESCRIPT_TARGETINFO_H_
