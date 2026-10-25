#include "TypeScript/CppMangler.h"

#include "llvm/TargetParser/Triple.h"

#include "gmock/gmock.h"

// @cppname's symbols, against the ones clang gives the same declarations
// (clang++ -target <triple> -S -emit-llvm): MSVC's scheme on windows-msvc, Itanium's elsewhere,
// and the <stdint.h> typedefs as each target defines them.
namespace
{

enum Kind
{
    Free,
    Method,
    Static
};

struct Case
{
    const char *declaration;
    Kind kind;
};

std::string mangle(const char *triple, const char *declaration, Kind kind = Free, llvm::StringRef className = {})
{
    CppMangleContext context;
    context.className = className;
    context.isInstanceMethod = kind == Method;
    context.isStaticMethod = kind == Static;
    std::string symbol, error;
    if (!mangleCppName(declaration, llvm::Triple(llvm::Triple::normalize(triple)), context, symbol, error))
    {
        return "error: " + error;
    }

    return symbol;
}

const Case cases[] = {
    {"int Methods::scaled(int)", Method},
    {"int Methods::scaled(double)", Method},
    {"Pair Methods::pair()", Method},
    {"int Methods::get() const", Method},
    {"int Methods::stat(int, int)", Static},
    {"void Methods::two(Pair*, Pair*)", Method},
    {"void Methods::refs(const Pair&, Pair&&, const Pair&)", Method},
    {"void Methods::ptrs(int*, const int*, int* const, const char*, void*)", Method},
    {"void Methods::self(Methods*, const Methods&)", Method},
    {"void Methods::all(bool, char, signed char, unsigned char, short, unsigned short, int, unsigned, long, unsigned long, "
     "long long, unsigned long long, float, double)",
     Method},
    {"void Methods::wide(wchar_t, char16_t, char32_t)", Method},
    {"void Methods::std(int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, size_t, ptrdiff_t, "
     "intptr_t, uintptr_t)",
     Method},
    {"void Methods::kinds(class Obj*, union U*, enum E, ns::Pt*, ns::in::Q*, ns::Pt*)", Method},
    {"void Methods::pp(int**, const char**, char* const*)", Method},
    {"void Methods::vol(volatile int*, const volatile int*)", Method},
    {"void ns::free(int)", Free},
    {"void ns::Pt::m(ns::Pt*, ns::in::Q*)", Method},
    {"int ns::in::deep(ns::Pt*, ns::in::Q*)", Free},
    {"void plain()", Free},
    {"void varf(int, ...)", Free},
    {"long long ret()", Free},
    {"const char* rets(int)", Free},
    {"void rep(const int*, const int*, double, double)", Free},
    {"Pair* retp(Pair)", Free},
    {"void many(Pair*, class Obj*, union U*, ns::Pt*, Methods*, const Pair*, const class Obj*, const union U*, "
     "const ns::Pt*, const Methods*, Pair*, Methods&, class Obj&)",
     Free},
    {"Pair* Methods::back(Pair*)", Method},
    {"long long* Methods::lb(long long*)", Method},
    {"int Methods::vf(int, ...)", Method},
    {"void rr(Pair&&, Pair&&)", Free},
    {"const char* const* cc(const char* const*)", Free},
};

const char *const msvcX64[] = {
    "?scaled@Methods@@QEAAHH@Z",
    "?scaled@Methods@@QEAAHN@Z",
    "?pair@Methods@@QEAA?AUPair@@XZ",
    "?get@Methods@@QEBAHXZ",
    "?stat@Methods@@SAHHH@Z",
    "?two@Methods@@QEAAXPEAUPair@@0@Z",
    "?refs@Methods@@QEAAXAEBUPair@@$$QEAU2@0@Z",
    "?ptrs@Methods@@QEAAXPEAHPEBHQEAHPEBDPEAX@Z",
    "?self@Methods@@QEAAXPEAU1@AEBU1@@Z",
    "?all@Methods@@QEAAX_NDCEFGHIJK_J_KMN@Z",
    "?wide@Methods@@QEAAX_W_S_U@Z",
    "?std@Methods@@QEAAXCEFGHI_J_K1001@Z",
    "?kinds@Methods@@QEAAXPEAVObj@@PEATU@@W4E@@PEAUPt@ns@@PEAUQ@in@6@3@Z",
    "?pp@Methods@@QEAAXPEAPEAHPEAPEBDPEBQEAD@Z",
    "?vol@Methods@@QEAAXPECHPEDH@Z",
    "?free@ns@@YAXH@Z",
    "?m@Pt@ns@@QEAAXPEAU12@PEAUQ@in@2@@Z",
    "?deep@in@ns@@YAHPEAUPt@2@PEAUQ@12@@Z",
    "?plain@@YAXXZ",
    "?varf@@YAXHZZ",
    "?ret@@YA_JXZ",
    "?rets@@YAPEBDH@Z",
    "?rep@@YAXPEBH0NN@Z",
    "?retp@@YAPEAUPair@@U1@@Z",
    "?many@@YAXPEAUPair@@PEAVObj@@PEATU@@PEAUPt@ns@@PEAUMethods@@PEBU1@PEBV2@PEBT3@PEBU45@PEBU6@0AEAU6@AEAV2@@Z",
    "?back@Methods@@QEAAPEAUPair@@PEAU2@@Z",
    "?lb@Methods@@QEAAPEA_JPEA_J@Z",
    "?vf@Methods@@QEAAHHZZ",
    "?rr@@YAX$$QEAUPair@@0@Z",
    "?cc@@YAPEBQEBDPEBQEBD@Z",
};

const char *const msvcX86[] = {
    "?scaled@Methods@@QAEHH@Z",
    "?scaled@Methods@@QAEHN@Z",
    "?pair@Methods@@QAE?AUPair@@XZ",
    "?get@Methods@@QBEHXZ",
    "?stat@Methods@@SAHHH@Z",
    "?two@Methods@@QAEXPAUPair@@0@Z",
    "?refs@Methods@@QAEXABUPair@@$$QAU2@0@Z",
    "?ptrs@Methods@@QAEXPAHPBHQAHPBDPAX@Z",
    "?self@Methods@@QAEXPAU1@ABU1@@Z",
    "?all@Methods@@QAEX_NDCEFGHIJK_J_KMN@Z",
    "?wide@Methods@@QAEX_W_S_U@Z",
    "?std@Methods@@QAEXCEFGHI_J_KIHHI@Z",
    "?kinds@Methods@@QAEXPAVObj@@PATU@@W4E@@PAUPt@ns@@PAUQ@in@6@3@Z",
    "?pp@Methods@@QAEXPAPAHPAPBDPBQAD@Z",
    "?vol@Methods@@QAEXPCHPDH@Z",
    "?free@ns@@YAXH@Z",
    "?m@Pt@ns@@QAEXPAU12@PAUQ@in@2@@Z",
    "?deep@in@ns@@YAHPAUPt@2@PAUQ@12@@Z",
    "?plain@@YAXXZ",
    "?varf@@YAXHZZ",
    "?ret@@YA_JXZ",
    "?rets@@YAPBDH@Z",
    "?rep@@YAXPBH0NN@Z",
    "?retp@@YAPAUPair@@U1@@Z",
    "?many@@YAXPAUPair@@PAVObj@@PATU@@PAUPt@ns@@PAUMethods@@PBU1@PBV2@PBT3@PBU45@PBU6@0AAU6@AAV2@@Z",
    "?back@Methods@@QAEPAUPair@@PAU2@@Z",
    "?lb@Methods@@QAEPA_JPA_J@Z",
    "?vf@Methods@@QAAHHZZ",
    "?rr@@YAX$$QAUPair@@0@Z",
    "?cc@@YAPBQBDPBQBD@Z",
};

// Itanium, with the typedefs of 64-bit Linux in `std`
const char *const itaniumLinux64[] = {
    "_ZN7Methods6scaledEi",
    "_ZN7Methods6scaledEd",
    "_ZN7Methods4pairEv",
    "_ZNK7Methods3getEv",
    "_ZN7Methods4statEii",
    "_ZN7Methods3twoEP4PairS1_",
    "_ZN7Methods4refsERK4PairOS0_S2_",
    "_ZN7Methods4ptrsEPiPKiS0_PKcPv",
    "_ZN7Methods4selfEPS_RKS_",
    "_ZN7Methods3allEbcahstijlmxyfd",
    "_ZN7Methods4wideEwDsDi",
    "_ZN7Methods3stdEahstijlmmllm",
    "_ZN7Methods5kindsEP3ObjP1U1EPN2ns2PtEPNS5_2in1QES7_",
    "_ZN7Methods2ppEPPiPPKcPKPc",
    "_ZN7Methods3volEPViPVKi",
    "_ZN2ns4freeEi",
    "_ZN2ns2Pt1mEPS0_PNS_2in1QE",
    "_ZN2ns2in4deepEPNS_2PtEPNS0_1QE",
    "_Z5plainv",
    "_Z4varfiz",
    "_Z3retv",
    "_Z4retsi",
    "_Z3repPKiS0_dd",
    "_Z4retp4Pair",
    "_Z4manyP4PairP3ObjP1UPN2ns2PtEP7MethodsPKS_PKS1_PKS3_PKS6_PKS8_S0_RS8_RS1_",
    "_ZN7Methods4backEP4Pair",
    "_ZN7Methods2lbEPx",
    "_ZN7Methods2vfEiz",
    "_Z2rrO4PairS0_",
    "_Z2ccPKPKc",
};

const size_t stdCase = 11;

void expectAll(const char *triple, const char *const (&expected)[std::size(cases)], const char *stdSymbol = nullptr)
{
    for (size_t i = 0; i < std::size(cases); ++i)
    {
        auto want = i == stdCase && stdSymbol ? stdSymbol : expected[i];
        EXPECT_EQ(mangle(triple, cases[i].declaration, cases[i].kind), want) << triple << ": " << cases[i].declaration;
    }
}

TEST(CppMangler, MsvcX64)
{
    expectAll("x86_64-pc-windows-msvc", msvcX64);
}

TEST(CppMangler, MsvcArm64IsMsvcX64)
{
    expectAll("aarch64-pc-windows-msvc", msvcX64);
}

TEST(CppMangler, MsvcX86)
{
    expectAll("i686-pc-windows-msvc", msvcX86);
}

TEST(CppMangler, ItaniumLinux64)
{
    expectAll("x86_64-unknown-linux-gnu", itaniumLinux64);
    expectAll("aarch64-unknown-linux-gnu", itaniumLinux64);
    expectAll("aarch64-unknown-linux-android29", itaniumLinux64);
}

// the same scheme; only the typedefs differ
TEST(CppMangler, ItaniumTypedefsByTarget)
{
    expectAll("x86_64-pc-windows-gnu", itaniumLinux64, "_ZN7Methods3stdEahstijxyyxxy");
    expectAll("i686-unknown-linux-gnu", itaniumLinux64, "_ZN7Methods3stdEahstijxyjiij");
    expectAll("x86_64-apple-darwin", itaniumLinux64, "_ZN7Methods3stdEahstijxymllm");
    expectAll("arm64-apple-darwin", itaniumLinux64, "_ZN7Methods3stdEahstijxymllm");

    const char *td = "void td(int64_t, size_t, ptrdiff_t, intptr_t, long)";
    EXPECT_EQ(mangle("wasm32-unknown-unknown", td), "_Z2tdxmlll");
    EXPECT_EQ(mangle("armv7-unknown-linux-gnueabihf", td), "_Z2tdxjiil");
    EXPECT_EQ(mangle("i686-apple-darwin", td), "_Z2tdxmill");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", td), "_Z2tdlmlll");
    EXPECT_EQ(mangle("x86_64-pc-windows-msvc", td), "?td@@YAX_J_K00J@Z");
    EXPECT_EQ(mangle("i686-pc-windows-msvc", td), "?td@@YAX_JIHHJ@Z");
}

// in a @cpp class an unqualified name is the class's method; parameter names and `std::` are
// what a C++ header has
TEST(CppMangler, ClassContextAndHeaderSpelling)
{
    EXPECT_EQ(mangle("x86_64-pc-windows-msvc", "int scaled(int x)", Method, "Methods"), "?scaled@Methods@@QEAAHH@Z");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", "int scaled(int x)", Method, "Methods"), "_ZN7Methods6scaledEi");
    EXPECT_EQ(mangle("x86_64-pc-windows-msvc", "struct Pair pair();", Method, "Base"), "?pair@Base@@QEAA?AUPair@@XZ");
    EXPECT_EQ(mangle("x86_64-pc-windows-msvc", "int stat(int, int)", Static, "Methods"), "?stat@Methods@@SAHHH@Z");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", "void f(std::size_t n, unsigned long int m, long double d)"), "_Z1fmme");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", "void f(void)"), "_Z1fv");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", "void f(...)"), "_Z1fz");
    EXPECT_EQ(mangle("x86_64-pc-windows-msvc", "void f(...)"), "?f@@YAXZZ");
}

// MSVC encodes a returned value's const, a builtin's as well as a record's; Itanium has no result
TEST(CppMangler, QualifiedResult)
{
    auto msvc = "x86_64-pc-windows-msvc";
    EXPECT_EQ(mangle(msvc, "const int f()"), "?f@@YA?BHXZ");
    EXPECT_EQ(mangle(msvc, "const struct P g()"), "?g@@YA?BUP@@XZ");
    EXPECT_EQ(mangle(msvc, "volatile int h()"), "?h@@YA?CHXZ");
    EXPECT_EQ(mangle(msvc, "const int* const k()"), "?k@@YAQEBHXZ");
    EXPECT_EQ(mangle("x86_64-unknown-linux-gnu", "const int f()"), "_Z1fv");
}

TEST(CppMangler, Errors)
{
    auto msvc = "x86_64-pc-windows-msvc";
    EXPECT_EQ(mangle(msvc, "int f(int"), "error: expected ')' at the end");
    EXPECT_EQ(mangle(msvc, "int f(int) const"), "error: only a method with 'this' can be const or volatile");
    EXPECT_EQ(mangle(msvc, "void f(std::vector<int>)"), "error: templates aren't supported at '<'");
    EXPECT_EQ(mangle(msvc, "int operator+(int)"), "error: operators, constructors and destructors aren't supported at 'operator'");
    EXPECT_EQ(mangle(msvc, "void ~A()"), "error: operators, constructors and destructors aren't supported at '~'");
    EXPECT_EQ(mangle(msvc, "void A::A()", Method), "error: constructors aren't supported");
    EXPECT_EQ(mangle(msvc, "int f()", Method), "error: the method's class isn't known: qualify the name, Class::f");
    EXPECT_EQ(mangle(msvc, "virtual int f()", Method), "error: a virtual method is called through its vtable slot, @slot(N), not by its symbol at 'virtual'");
    EXPECT_EQ(mangle(msvc, "void f(int (*)(int))"), "error: function pointers and arrays aren't supported at '('");
    EXPECT_EQ(mangle(msvc, "void f(int = 0)"), "error: default arguments aren't supported at '='");
    EXPECT_EQ(mangle(msvc, "void f(int, void)"), "error: a parameter can't be void at ')'");
    EXPECT_EQ(mangle(msvc, "unsigned double f()"), "error: invalid combination of type specifiers at 'f'");
    EXPECT_EQ(mangle(msvc, "int f() noexcept"), "error: ref-qualifiers and exception specifications aren't supported at 'noexcept'");
    EXPECT_EQ(mangle(msvc, "f()"), "error: expected a name at '('");
}

} // namespace
