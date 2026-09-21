#include "TypeScript/TargetInfo.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "llvm/TargetParser/Triple.h"

#include "mlir/IR/MLIRContext.h"

#include "gmock/gmock.h"

// TargetInfo replaces a hand-maintained list of 64-bit architectures in opts.cpp; any arch not on
// it got 32. The list was wrong in two ways: it put aarch64_32 among the 64-bit arches (it is ILP32,
// see Arm64_32IsAnIlp32Target), and it missed 64-bit arches - systemz, sparcv9, amdgcn, ve, spirv,
// riscv64be - which therefore got 32. These tests cover arches nobody has built for, to prove the
// answer comes from the triple rather than from somebody remembering to add a line.
namespace
{

llvm::Triple t(const char *str)
{
    return llvm::Triple(llvm::Triple::normalize(str));
}

const llvm::Triple &hostX64()
{
    static llvm::Triple host = t("x86_64-pc-windows-msvc");
    return host;
}

TargetInfo infoFor(const char *target)
{
    return TargetInfo::fromTriple(t(target), hostX64());
}

TEST(TargetInfoTest, PointerWidthComesFromTheTriple)
{
    EXPECT_EQ(infoFor("x86_64-pc-windows-msvc").pointerBits, 64u);
    EXPECT_EQ(infoFor("i686-pc-windows-msvc").pointerBits, 32u);
    EXPECT_EQ(infoFor("wasm32-unknown-unknown").pointerBits, 32u);
    EXPECT_EQ(infoFor("aarch64-unknown-linux-gnu").pointerBits, 64u);
}

// 64-bit arches missing from the old list, which gave them 32. Each spelling is one that
// llvm::Triple's parseArch maps to the arch named in the comment.
TEST(TargetInfoTest, PointerWidthIsRightForArchesTheOldListMissed)
{
    EXPECT_EQ(infoFor("s390x-unknown-linux-gnu").pointerBits, 64u); // systemz
    EXPECT_EQ(infoFor("sparcv9-sun-solaris").pointerBits, 64u);     // sparcv9
    EXPECT_EQ(infoFor("amdgcn-amd-amdhsa").pointerBits, 64u);       // amdgcn
    EXPECT_EQ(infoFor("ve-unknown-linux-gnu").pointerBits, 64u);    // ve
}

// An unknown arch has no width of its own; falling back to 0 would make every size computation
// degenerate, so it takes the host's.
TEST(TargetInfoTest, UnknownArchFallsBackToHostWidth)
{
    EXPECT_EQ(infoFor("unknown-unknown-unknown").pointerBits, 64u);
}

// MSVC C++ EH cross-references are image-base-relative RVAs on 64-bit Windows and absolute
// pointers on 32-bit Windows (x86 and ARM). Non-Windows targets use the Itanium scheme and neither applies.
TEST(TargetInfoTest, ImageBaseRelativeEHIsWin64Only)
{
    EXPECT_TRUE(infoFor("x86_64-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_TRUE(infoFor("aarch64-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("i686-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("thumbv7-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("x86_64-unknown-linux-gnu").usesImageBaseRelativeEH);
}

// _CxxThrowException is __stdcall on 32-bit x86 only, which is what puts the @8 on the symbol.
TEST(TargetInfoTest, StdcallThrowIsWin32X86Only)
{
    EXPECT_TRUE(infoFor("i686-pc-windows-msvc").stdcallDecoratesCxxThrow);
    EXPECT_FALSE(infoFor("x86_64-pc-windows-msvc").stdcallDecoratesCxxThrow);
    EXPECT_FALSE(infoFor("i686-unknown-linux-gnu").stdcallDecoratesCxxThrow);
}

// tslang.exe is an x64 process and cannot execute i386 or wasm in-process.
TEST(TargetInfoTest, InProcessJitNeedsAMatchingHost)
{
    EXPECT_TRUE(infoFor("x86_64-pc-windows-msvc").supportsInProcessJit);
    EXPECT_FALSE(infoFor("i686-pc-windows-msvc").supportsInProcessJit);
    EXPECT_FALSE(infoFor("wasm32-unknown-unknown").supportsInProcessJit);
    EXPECT_FALSE(infoFor("x86_64-unknown-linux-gnu").supportsInProcessJit);
}

// sizeBits is what ~20 existing call sites read. It must keep answering, and it must now answer
// from TargetInfo rather than from a separate field that could drift away from it.
TEST(TargetInfoTest, CompileOptionsSizeBitsFollowsTargetInfo)
{
    CompileOptions options;

    options.targetInfo = infoFor("i686-pc-windows-msvc");
    EXPECT_EQ(options.sizeBits(), 32);

    options.targetInfo = infoFor("x86_64-pc-windows-msvc");
    EXPECT_EQ(options.sizeBits(), 64);
}

// TypeHelper is constructed at 152 sites, nearly all of which only ask for width-independent
// types. Rather than thread CompileOptions through all of them, the options are optional and
// only the sites that need a width pass them - so getSizeType() has to answer from the target,
// and has to refuse rather than guess when nobody supplied one.
TEST(TargetInfoTest, TypeHelperSizeTypeFollowsTheTarget)
{
    mlir::MLIRContext context;

    CompileOptions options32;
    options32.targetInfo = infoFor("i686-pc-windows-msvc");
    EXPECT_EQ(::typescript::TypeHelper(&context, options32).getSizeType(),
              mlir::IntegerType::get(&context, 32));

    CompileOptions options64;
    options64.targetInfo = infoFor("x86_64-pc-windows-msvc");
    EXPECT_EQ(::typescript::TypeHelper(&context, options64).getSizeType(),
              mlir::IntegerType::get(&context, 64));
}

// ARM64_32 (watchOS) is the arch the old hand-written list in opts.cpp got wrong: it listed
// aarch64_32 among the 64-bit arches, but ILP32 means 32-bit pointers. Deriving the width from
// the triple fixes that, and this test is what keeps it fixed.
TEST(TargetInfoTest, Arm64_32IsAnIlp32Target)
{
    EXPECT_EQ(infoFor("arm64_32-apple-watchos").pointerBits, 32u);
}

} // namespace
