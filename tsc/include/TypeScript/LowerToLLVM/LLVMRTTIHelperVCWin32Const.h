#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_

namespace typescript
{

constexpr const auto *typeInfoExtRef = "??_7type_info@@6B@";
constexpr const auto *imageBaseRef = "__ImageBase";

namespace F32Type
{
constexpr const auto *typeName = ".N";
constexpr const auto *typeInfoRef = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0N@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1N";
constexpr const auto *throwInfoRef = "_TI1N";
} // namespace F32Type

namespace F64Type
{
constexpr const auto *typeName = ".N";
constexpr const auto *typeInfoRef = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0N@88";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1N";
constexpr const auto *throwInfoRef = "_TI1N";
} // namespace F64Type

namespace I32Type
{
constexpr const auto *typeName = ".H";
constexpr const auto *typeInfoRef = "??_R0H@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0H@84";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA1H";
constexpr const auto *throwInfoRef = "_TI1H";
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

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
