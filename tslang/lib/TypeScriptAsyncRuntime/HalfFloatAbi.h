// How LLVM passes an `f16` to and from compiler-rt's half conversions (compiler-rt/README.md), in C
// any compiler takes - MSVC has no `_Float16`.
//
// x86-64 and AArch64: in the first floating-point register, its low 16 bits (XMM0; H0, the low half
// of S0) - where a `float` goes, so the half's bits travel as a `float`'s low 16 bits. 32-bit x86 and
// the rest: in an integer register or on the stack, as a `uint16_t`.
#ifndef TSLANG_HALF_FLOAT_ABI_H
#define TSLANG_HALF_FLOAT_ABI_H

#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

typedef float tslang_half_t;

static inline tslang_half_t tslang_half_from_bits(uint16_t bits)
{
    uint32_t word = bits;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static inline uint16_t tslang_half_to_bits(tslang_half_t half)
{
    uint32_t word;
    memcpy(&word, &half, sizeof(word));
    return (uint16_t)word;
}

#else

typedef uint16_t tslang_half_t;

static inline tslang_half_t tslang_half_from_bits(uint16_t bits)
{
    return bits;
}

static inline uint16_t tslang_half_to_bits(tslang_half_t half)
{
    return half;
}

#endif

#endif // TSLANG_HALF_FLOAT_ABI_H
