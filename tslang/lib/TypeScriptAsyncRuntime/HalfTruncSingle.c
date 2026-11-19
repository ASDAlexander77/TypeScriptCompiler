// float -> half, as compiler-rt's truncsfhf2.c, in the half ABI LLVM calls it with (HalfFloatAbi.h)
#define SRC_SINGLE
#define DST_HALF
#include "compiler-rt/fp_trunc_impl.inc"
#include "HalfFloatAbi.h"

tslang_half_t __truncsfhf2(float a)
{
    return tslang_half_from_bits(__truncXfYf2__(a));
}
