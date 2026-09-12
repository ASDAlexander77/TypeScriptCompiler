#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_

namespace typescript
{

namespace windows
{

// NOTE: every mangled name below is 64-bit MSVC mangling - `PEA` is a `__ptr64` pointer, and the
// trailing digits of a `_CT...` name are the size of the value a catch copies. A 32-bit target
// needs its own table (`PA`, and `@84` for pointers), not just a different size; nothing here
// adapts on its own.
//
// `catchableTypeSize` is what goes in CatchableType::sizeOrOffset, and it is load-bearing: the
// CRT copies exactly that many bytes into the catch variable's frame slot, so a size that is too
// large overwrites whatever the frame put above that slot. It must agree with the digits in
// `catchableTypeInfoRef`. Pointer-shaped types take the target's pointer size instead of a
// constant here, so they are not listed.

constexpr const auto *typeInfoExtRef = "??_7type_info@@6B@";
constexpr const auto *imageBaseRef = "__ImageBase";

namespace F32Type
{
constexpr const auto *typeName = ".N";
constexpr const auto *typeInfoRef = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0N@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1N";
constexpr const auto *throwInfoRef = "_TI1N";
// describes `.N` (double), like F64Type - see setF32AsCatchType
constexpr int catchableTypeSize = 8;
} // namespace F32Type

namespace F64Type
{
constexpr const auto *typeName = ".N";
constexpr const auto *typeInfoRef = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0N@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1N";
constexpr const auto *throwInfoRef = "_TI1N";
constexpr int catchableTypeSize = 8;
} // namespace F64Type

namespace I32Type
{
constexpr const auto *typeName = ".H";
constexpr const auto *typeInfoRef = "??_R0H@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0H@84";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1H";
constexpr const auto *throwInfoRef = "_TI1H";
// 4, not the pointer size: `int` is 4 bytes on every target, and the `4` at the end of
// `_CT??_R0H@84` says so too
constexpr int catchableTypeSize = 4;
} // namespace I32Type

namespace StringType
{
constexpr const auto *typeName = ".PEAD";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0PEAD@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0PEAD@88";
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2PEAD";
constexpr const auto *throwInfoRef = "_TIC2PEAD";
} // namespace StringType

namespace I8PtrType
{
constexpr const auto *typeName = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0PEAX@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1PEAX";
constexpr const auto *throwInfoRef = "_TIC1PEAX";
} // namespace I8PtrType

namespace ClassType
{
constexpr const auto *typeName = ".PEAV";
constexpr const auto *typeNameSuffix = "@@";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0PEAV";
constexpr const auto *typeInfoRefSuffix = "@@@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0PEAV";
constexpr const auto *catchableTypeInfoRefSuffix = "@@@88";
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2PEAV";
constexpr const auto *catchableTypeInfoArrayRefSuffix = "@@";
constexpr const auto *throwInfoRef = "_TI2PEAV";
constexpr const auto *throwInfoRefSuffix = "@@";
} // namespace ClassType

} // namespace windows

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
