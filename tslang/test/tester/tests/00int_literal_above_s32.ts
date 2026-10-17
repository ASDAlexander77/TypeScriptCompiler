// regression test: an integer literal that does not fit s32 gets a 64-bit type.
//
// bug found and fixed: a literal in [2^31, 2^32) (3000000000, 2147483648, 0xFFFFFFFF) was typed u32,
// so arithmetic with other integer literals was done in 32 bits: `1 + 3000000000` printed
// -1294967295. Such a literal is now s64, like 5000000000 already was; smaller literals stay s32.

function main() {
    const x = 3000000000;
    print(x);
    assert(`${x}` == "3000000000", "const");

    const sum = 1 + 3000000000;
    print(sum);
    assert(`${sum}` == "3000000001", "1 + 3000000000");

    const twice = 3000000000 * 2;
    assert(`${twice}` == "6000000000", "3000000000 * 2");

    let z = 3000000000;
    z += 3000000000;
    assert(`${z}` == "6000000000", "z += 3000000000");

    const allOnes = 0xFFFFFFFF + 1;
    assert(`${allOnes}` == "4294967296", "0xFFFFFFFF + 1");

    const belowMin = -2147483649;
    assert(`${belowMin}` == "-2147483649", "-2147483649");

    const minPlus = 2147483648 + 1;
    assert(`${minPlus}` == "2147483649", "2147483648 + 1");

    // a literal that fits s32 is still s32
    const small = 2147483647;
    assert(`${small}` == "2147483647", "s32 max");

    // narrowing into a declared 32-bit type still works
    const m: s32 = -2147483648;
    assert(`${m}` == "-2147483648", "s32 min");
    const u: u32 = 0xFFFFFFFF;
    assert(`${u}` == "4294967295", "u32 max");
    const v: u32 = 3000000000;
    assert(`${v}` == "3000000000", "u32 3000000000");

    print("done.");
}
