// `<Opaque><any>0` read 8 bytes out of a box holding a 4-byte s32: the padding after it became the
// pointer's high bytes, garbage wherever the allocator does not zero memory (-mm=rc, none, own), so
// the "null" pointer was not null. An Opaque out of an `any` is as wide as the value in the box.

function opaqueOf(value: any): Opaque {
    return <Opaque>value;
}

function isNull(p: Opaque): boolean {
    return <u64><index>p == 0;
}

// leaves freed blocks of a box's size filled with ones, so a box made next is not zeroed by luck
// (where blocks are freed: rc, own)
function dirty() {
    for (let i = 0; i < 4; i++) {
        const junk: u64[] = [];
        for (let j = 0; j < 4; j++) junk.push(0xFFFFFFFFFFFFFFFF);
    }
}

function main() {
    for (let i = 0; i < 100; i++) {
        dirty();
        const none: Opaque = <Opaque><any>0;
        assert(isNull(none), "an s32 0 in an any is a null Opaque");
    }

    assert(isNull(opaqueOf(<s32>0)), "through a parameter");
    assert(<u64><index>opaqueOf(<s32>0x1234) == 0x1234, "an s32 keeps its value");
    assert(<u64><index>opaqueOf(<u8>0xAB) == 0xAB, "a u8");
    assert(<u64><index>opaqueOf(<u16>0xABCD) == 0xABCD, "a u16");
    assert(<u64><index>opaqueOf(<u64>0x123456789A) == 0x123456789A, "a u64 is all 8 bytes");

    print("done.");
}
