#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_

namespace typescript
{

namespace linux
{

// `float` - it used to be `_ZTId`, so a thrown f32 was read as a double
namespace F32Type
{
constexpr const auto *typeName = "_ZTIf";
} // namespace F32Type

namespace F64Type
{
constexpr const auto *typeName = "_ZTId";
} // namespace F64Type

namespace I32Type
{
constexpr const auto *typeName = "_ZTIi";
} // namespace I32Type

// `unsigned int` - a u32 used to be thrown as an `int`, and read back with its top bit as a sign
namespace U32Type
{
constexpr const auto *typeName = "_ZTIj";
} // namespace U32Type

namespace BoolType
{
constexpr const auto *typeName = "_ZTIb";
} // namespace BoolType

// also a bigint, which is an i64
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
// A class is thrown as a pointer, and its `_ZTIP<n><name>` - a __pointer_type_info - has one
// more field after the four libstdc++ reads: the address of a thunk that boxes the thrown
// instance into an `any` (the class's own descriptor included). An untyped catch calls it for
// a class it knows nothing about, e.g. one only another module throws - the Itanium
// counterpart of the Windows `.PEAX` copy thunk. Every module emitting the type_info (a throw
// or a typed catch of the class) emits the same thunk, so the linkonce_odr copies agree.
constexpr const auto *boxThunkPrefix = ".eh.copy.box.";
constexpr int boxThunkField = 4;
// Only tslang's own __pointer_type_info has that field, so it says so in __flags - a bit
// libstdc++ leaves alone (it defines the low qualifier bits only, and compares a throw's flags
// with a catch's, which are emitted here alike). Any other pointer exception - a C++
// `throw "text"` (`_ZTIPKc`), a pointer to a C++ class, a type_info from before the field -
// ends after the pointee, and reading the field there called whatever came next.
constexpr int flagsField = 2;
constexpr int boxThunkFlag = 0x40000000;
} // namespace ClassType

} // namespace linux

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUXCONST_H_
