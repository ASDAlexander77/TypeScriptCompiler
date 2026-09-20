#include "TypeScript/TargetInfo.h"
#include "TypeScript/DataStructs.h"

#include "llvm/TargetParser/Triple.h"

#include "gmock/gmock.h"

// TargetInfo replaces a hand-maintained list of 64-bit architectures in opts.cpp. That list had
// already rotted - loongarch64 was on it, riscv32 appeared nowhere - so these tests deliberately
// cover arches nobody has built for, to prove the answer comes from the triple rather than from
// somebody remembering to add a line.
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

// The arches the old list got wrong, in both directions.
TEST(TargetInfoTest, PointerWidthIsRightForArchesTheOldListMissed)
{
    EXPECT_EQ(infoFor("riscv32-unknown-elf").pointerBits, 32u);
    EXPECT_EQ(infoFor("armv7-unknown-linux-gnueabihf").pointerBits, 32u);
    EXPECT_EQ(infoFor("loongarch64-unknown-linux-gnu").pointerBits, 64u);
}

// An unknown arch has no width of its own; falling back to 0 would make every size computation
// degenerate, so it takes the host's.
TEST(TargetInfoTest, UnknownArchFallsBackToHostWidth)
{
    EXPECT_EQ(infoFor("unknown-unknown-unknown").pointerBits, 64u);
}

// MSVC C++ EH cross-references are image-base-relative RVAs on 64-bit Windows and absolute
// pointers on 32-bit x86. Non-Windows targets use the Itanium scheme and neither applies.
TEST(TargetInfoTest, ImageBaseRelativeEHIsWin64Only)
{
    EXPECT_TRUE(infoFor("x86_64-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("i686-pc-windows-msvc").usesImageBaseRelativeEH);
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

} // namespace
