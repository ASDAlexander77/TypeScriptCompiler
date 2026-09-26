// C expects 8- and 16-bit integers extended to 32 bits by whoever hands them over: the caller
// for an argument, the callee for a result (clang relies on both on x86-64 SysV).

declare function take_s8(v: s8): s32;
declare function take_s16(v: s16): s32;
declare function take_u8(v: u8): s32;
declare function take_u16(v: u16): s32;
declare function take_bool(v: boolean): s32;
declare function take_s32(v: s32): s32;
declare function give_s8(): s8;
declare function give_u16(): u16;

function main() {
    print(take_s8(-5), take_s16(-3), take_u8(255), take_u16(65535), take_bool(true), take_s32(1), give_s8(), give_u16());
}
