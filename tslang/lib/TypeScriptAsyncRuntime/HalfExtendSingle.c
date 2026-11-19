// half -> float, as compiler-rt's extendhfsf2.c, in the half ABI LLVM calls it with (HalfFloatAbi.h)
#define SRC_HALF
#define DST_SINGLE
#include "compiler-rt/fp_extend_impl.inc"
#include "HalfFloatAbi.h"

float __extendhfsf2(tslang_half_t a)
{
    return __extendXfYf2__(tslang_half_to_bits(a));
}
