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

    // MSVC C++ member functions are __thiscall on 32-bit x86 (`this` in ECX), not the plain C
    // convention a class method is called with, so a @cppvtable class can't call them.
    bool memberCallsUseThiscall = false;

    // MSVC returns a C++ record (struct, class) from an instance method through a hidden pointer,
    // passed after `this` and returned in turn (clang's MicrosoftCXXABI::classifyReturnType): the
    // convention of a @cppvtable method declared to return a tuple. The Itanium ABI passes it
    // before `this`, and only for some records.
    bool hiddenReturnPointerAfterThis = false;

    // The Itanium ABI on x86-64 returns a record with a destructor or a copy constructor (marked
    // @cppnontrivial) through a hidden pointer passed before `this` - an ordinary first argument,
    // in RDI - and returned in turn; a record of one or two 8-byte integer or pointer fields
    // without them comes back in RAX:RDX, as a tuple is returned. (AArch64 passes the hidden
    // pointer in X8, which an ordinary argument isn't, so it has neither.)
    bool hiddenReturnPointerBeforeThis = false;

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

        // By pointer width, as clang's MicrosoftCXXABI::isImageRelative() decides it: 32-bit ARM
        // Windows stores absolute pointers too, not just x86.
        info.usesImageBaseRelativeEH = msvc && info.pointerBits == 64;
        info.stdcallDecoratesCxxThrow = msvc && x86_32;
        info.supportsInProcessJit =
            target.getArch() == host.getArch() && target.getOS() == host.getOS();
        info.memberCallsUseThiscall = msvc && x86_32;
        info.hiddenReturnPointerAfterThis = msvc;
        info.hiddenReturnPointerBeforeThis = !msvc && target.getArch() == llvm::Triple::x86_64;

        return info;
    }
};

#endif // TYPESCRIPT_TARGETINFO_H_
