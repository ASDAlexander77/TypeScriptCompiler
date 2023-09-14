#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_

namespace typescript
{

namespace linux
{

namespace F32Type
{
constexpr const auto *typeName = "_ZTId";
} // namespace F32Type

namespace F64Type
{
constexpr const auto *typeName = "_ZTId";
} // namespace F64Type

namespace I32Type
{
constexpr const auto *typeName = "_ZTIi";
} // namespace I32Type

namespace I64Type
{
constexpr const auto *typeName = "_ZTIx";
} // namespace I64Type

namespace ConstStringType
{
constexpr const auto *typeName = "_ZTIPKc";
} // namespace ConstStringType

namespace StringType
{
constexpr const auto *typeName = "_ZTIPc";
} // namespace StringType

namespace I8PtrType
{
constexpr const auto *typeName = "_ZTIPv";
} // namespace I8PtrType

namespace ClassType
{
constexpr const auto *typeName = "";
constexpr const auto *classTypeInfoName = "_ZTVN10__cxxabiv117__class_type_infoE";
constexpr const auto *singleInheritanceClassTypeInfoName = "_ZTVN10__cxxabiv120__si_class_type_infoE";
constexpr const auto *pointerTypeInfoName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
} // namespace ClassType

} // namespace linux

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
