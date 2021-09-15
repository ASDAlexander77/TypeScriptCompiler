#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_

namespace typescript
{
namespace F32Type
{
constexpr auto typeName = ".N";
constexpr auto typeInfoRef = "??_R0N@8";
constexpr auto catchableTypeInfoRef = "_CT??_R0N@88";
constexpr auto catchableTypeInfoArrayRef = "_CTA1N";
constexpr auto throwInfoRef = "_TI1N";
} // namespace F32Type

namespace I32Type
{
constexpr auto typeName = ".H";
constexpr auto typeInfoRef = "??_R0H@8";
constexpr auto catchableTypeInfoRef = "_CT??_R0H@84";
constexpr auto catchableTypeInfoArrayRef = "_CTA1H";
constexpr auto throwInfoRef = "_TI1H";
} // namespace I32Type

namespace I8PtrType
{
constexpr auto typeName = ".PEAD";
constexpr auto typeInfoRef = "??_R0PEAD@8";
constexpr auto typeInfoRef2 = "??_R0PEAX@88";
constexpr auto catchableTypeInfoRef = "_CT??_R0PEAD@88";
constexpr auto catchableTypeInfoRef2 = "_CT??_R0PEAX@88";
constexpr auto catchableTypeInfoArrayRef = "_CTA2PEAD";
constexpr auto throwInfoRef = "_TIC2PEAD";
} // namespace I8PtrType

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
