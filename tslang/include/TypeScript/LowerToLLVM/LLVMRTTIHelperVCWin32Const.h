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

// F32Type/F64Type/I32Type each also carry a second, `.PEAX` (void*) catchable-type entry -
// see the comment on StringType::typeName2 below for why: it is what lets an untyped
// `catch (e)` or a `catch (e: any)` (which filters on `??_R0PEAX@8`, the same generic
// void*-shaped RTTI used for `any`) match a thrown primitive at all. Without it the
// CatchableTypeArray the CRT walks contains only the primitive's own descriptor, the
// filter never matches, and the exception propagates out of `main` uncaught - see
// throw-number-catchable-type-missing-any-fallback.md.
//
// That fallback entry's *type descriptor* (`typeInfoRef2`, `??_R0PEAX@8`) is safe to share
// with String/Class/I8PtrType below: a type descriptor carries no size, just a vtable pointer
// and the `.PEAX` name. Its *catchable-type* record is not safe to share, though - it carries
// `sizeOrOffset`, and that value is genuinely different per primitive (4 for an `int`, 8 for a
// `double`, `pointerSize()` - 4 or 8 depending on target - for a string/class/`any` pointer).
// `_CT??_R0PEAX@88` is a `linkonce_odr` global keyed only by name: whichever translation unit
// or, within one module, whichever type gets processed first "wins" and fixes that symbol's
// size for every later reference to the same name (see `catchableType()`'s
// `lookupSymbol(...) -> return failure()` idempotency guard). Giving `I32Type` and
// `F32Type`/`F64Type` their own distinctly-named fallback records below keeps a module that
// throws both an `int` and a `string` (say) from having one of them silently take on the
// other's copy size.

// Copy-function records. A CatchableType whose properties do not say "simple type" and that
// names a copyFunction is not memcpy'd into the catch variable: the CRT calls
// copyFunction(catchVariable, thrownObject) instead (`__thiscall` on x86 - CxxThrowCallingConvPass
// gives every function named with copyThunkPrefix that convention). That is what lets a thrown
// value reach a handler of a different shape:
//  - the `.PEAX` entry of an int/double/string/class throw boxes the value into an `any`, which
//    is what an untyped `catch (e)` or `catch (e: any)` binds (they filter on `??_R0PEAX@8`).
//    It used to be a plain copy, so the catch variable held the raw value and the first read of
//    it as an `any` crashed;
//  - an int throw also lists `.N` with a thunk widening it to a double, so `catch (e: number)`
//    catches `throw 1` (an integer literal is an `int` here).
// These records, and the arrays and ThrowInfos that list them, have names of their own: an
// object built before them carries linkonce_odr records under the old names with the old
// contents, and the linker keeps whichever copy it sees first.
constexpr const auto *copyThunkPrefix = ".eh.copy.";
constexpr int copyFunctionProperties = 0;
constexpr int simpleTypeProperties = 1;

// `float`. It used to describe itself as `.N` (double) with a size of 8 - the F64Type records
// under the F64Type names - so `catch (e: number)` caught a thrown f32 and read 8 bytes of a
// 4-byte value, and a module throwing both kept whichever thunk came first.
namespace F32Type
{
constexpr const auto *typeName = ".M";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0M@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0M@84";
// own record, not the pointer-shaped `_CT??_R0PEAX@88` - see the block comment above
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@M84.box";
constexpr const auto *copyThunk2 = ".eh.copy.box.M";
// the third entry: `.N` (double), reached through a widening thunk, as for an int
constexpr const auto *typeName3 = ".N";
constexpr const auto *typeInfoRef3 = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef3 = "_CT??_R0N@88.fromM";
constexpr const auto *copyThunk3 = ".eh.copy.num.M";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA3M";
constexpr const auto *throwInfoRef = "_TI3M";
constexpr int catchableTypeSize = 4;
} // namespace F32Type

namespace F64Type
{
constexpr const auto *typeName = ".N";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0N@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0N@88";
// own record, not the pointer-shaped `_CT??_R0PEAX@88` - see the block comment above
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@N88.box";
constexpr const auto *copyThunk2 = ".eh.copy.box.N";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2N.box";
constexpr const auto *throwInfoRef = "_TI2N.box";
constexpr int catchableTypeSize = 8;
} // namespace F64Type

namespace I32Type
{
constexpr const auto *typeName = ".H";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0H@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0H@84";
// own record, not the pointer-shaped `_CT??_R0PEAX@88` - see the block comment above
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@84.box";
constexpr const auto *copyThunk2 = ".eh.copy.box.H";
// the third entry: `.N` (double), reached through a widening thunk
constexpr const auto *typeName3 = ".N";
constexpr const auto *typeInfoRef3 = "??_R0N@8";
constexpr const auto *catchableTypeInfoRef3 = "_CT??_R0N@88.fromH";
constexpr const auto *copyThunk3 = ".eh.copy.num.H";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA3H";
constexpr const auto *throwInfoRef = "_TI3H";
// 4, not the pointer size: `int` is 4 bytes on every target, and the `4` at the end of
// `_CT??_R0H@84` says so too
constexpr int catchableTypeSize = 4;
} // namespace I32Type

// `bool` - a boolean is an i1, stored as a byte
namespace BoolType
{
constexpr const auto *typeName = "._N";
constexpr const auto *typeInfoRef = "??_R0_N@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0_N@81";
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@_N81.box";
constexpr const auto *copyThunk2 = ".eh.copy.box._N";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2_N.box";
constexpr const auto *throwInfoRef = "_TI2_N.box";
constexpr int catchableTypeSize = 1;
} // namespace BoolType

// `__int64` - a bigint is an i64
namespace BigIntType
{
constexpr const auto *typeName = "._J";
constexpr const auto *typeInfoRef = "??_R0_J@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0_J@88";
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@_J88.box";
constexpr const auto *copyThunk2 = ".eh.copy.box._J";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2_J.box";
constexpr const auto *throwInfoRef = "_TI2_J.box";
constexpr int catchableTypeSize = 8;
} // namespace BigIntType

namespace StringType
{
constexpr const auto *typeName = ".PEAD";
constexpr const auto *typeName2 = ".PEAX";
constexpr const auto *typeInfoRef = "??_R0PEAD@8";
constexpr const auto *typeInfoRef2 = "??_R0PEAX@8";
constexpr const auto *catchableTypeInfoRef = "_CT??_R0PEAD@88";
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@88.box.PEAD";
constexpr const auto *copyThunk2 = ".eh.copy.box.PEAD";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2PEAD.box";
constexpr const auto *throwInfoRef = "_TIC2PEAD.box";
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
// per class, like the thunk it names: catchableTypeInfoRef2 + name + suffix
constexpr const auto *catchableTypeInfoRef2 = "_CT??_R0PEAX@88.box.PEAV";
constexpr const auto *catchableTypeInfoRef2Suffix = "@@";
constexpr const auto *copyThunk2 = ".eh.copy.box.PEAV";
constexpr const auto *copyThunk2Suffix = "@@";
constexpr const auto *catchableTypeInfoArrayRef = "_CTA2PEAV";
constexpr const auto *catchableTypeInfoArrayRefSuffix = "@@.box";
constexpr const auto *throwInfoRef = "_TI2PEAV";
constexpr const auto *throwInfoRefSuffix = "@@.box";
} // namespace ClassType

} // namespace windows

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32CONST_H_
