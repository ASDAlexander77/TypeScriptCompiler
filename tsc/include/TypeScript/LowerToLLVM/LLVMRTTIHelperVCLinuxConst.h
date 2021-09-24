#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_

namespace typescript
{
namespace F32Type
{
constexpr auto typeName = "_ZTId";
} // namespace F32Type

namespace I32Type
{
constexpr auto typeName = "_ZTIi";
} // namespace I32Type

namespace ConstStringType
{
constexpr auto typeName = "_ZTIPKc";
} // namespace ConstStringType

namespace StringType
{
constexpr auto typeName = "_ZTIPc";
} // namespace StringType

namespace I8PtrType
{
constexpr auto typeName = "_ZTIPv";
} // namespace I8PtrType

namespace ClassType
{
constexpr auto typeName = "";
constexpr auto classTypeInfoName = "_ZTVN10__cxxabiv117__class_type_infoE";
constexpr auto singleInheritanceClassTypeInfoName = "_ZTVN10__cxxabiv120__si_class_type_infoE";
constexpr auto pointerTypeInfoName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
} // namespace ClassType

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
