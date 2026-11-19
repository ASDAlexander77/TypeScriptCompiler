# compiler-rt builtins: half-precision conversions

LLVM lowers a conversion to or from `f16` to a call: `__truncsfhf2` and `__truncdfhf2` (float and
double to half), `__extendhfsf2` and `__extendhfdf2` (half to float and double). These functions are
compiler-rt's (and libgcc's, from GCC 12). MSVC's runtime has none of them, so a program that
converts an `f16` did not link on Windows.

The files here are compiler-rt's own, copied unchanged from
`llvm-project/compiler-rt/lib/builtins` (commit ca7933e47d3a3451d81e72ac174dcb5aa28b59d1), under its
license (LICENSE.TXT, Apache-2.0 WITH LLVM-exception):

- `fp_trunc.h`, `fp_trunc_impl.inc`: wider float to narrower
- `fp_extend.h`, `fp_extend_impl.inc`: narrower float to wider
- `int_lib.h`, `int_types.h`, `int_util.h`, `int_endianness.h`: what those include

The four entry points are in `../HalfTrunc*.c` and `../HalfExtend*.c`, not compiler-rt's
`truncsfhf2.c` and the rest: compiler-rt passes a half as `_Float16` when the compiler has one and as
`uint16_t` otherwise, and MSVC has no `_Float16`. LLVM passes it where `_Float16` goes, which on
x86-64 and AArch64 is a floating-point register, not an integer one (`HalfFloatAbi.h`).
