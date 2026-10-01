// a sign-neutral iN is signed: it prints, widens and compares keeping its sign (`let a: i32 = -1` printed
// 4294967295, `let c: i64 = -7` held 4294967289, `i16` -300 compared as 65236); a uN stays unsigned
function main() {
    let a: i32 = -1;
    assert(`${a}` == "-1", "i32 prints signed");
    assert(a < 0, "i32 compares signed");
    const n: number = a;
    assert(n == -1, "i32 to number");

    let b: i8 = -5;
    assert(`${b}` == "-5", "i8 prints signed");

    let c: i64 = -7;
    assert(`${c}` == "-7", "i32 literal widens to i64 with its sign");
    assert(c < 0, "i64 compares signed");

    let d: i16 = -300;
    assert(d < 0, "i16 widens to compare with its sign");
    assert(`${d}` == "-300", "i16 prints signed");

    const x: any = a;
    assert(`${x}` == "-1", "i32 in any");

    let k: i32 = -2;
    let m: index = k;
    assert(`${m}` == "-2", "i32 widens to index with its sign");

    let big: i64 = -9000000000;
    assert(`${big}` == "-9000000000" && big < 0, "i64");

    // unsigned stays unsigned
    let u: u32 = 4294967295;
    assert(`${u}` == "4294967295", "u32 prints unsigned");
    let u8v: u8 = 255;
    let widened: s32 = u8v;
    assert(widened == 255, "u8 widens with zeros");
    let e: i64 = u;
    assert(`${e}` == "4294967295", "u32 widens to i64 with zeros");

    print("done.");
}
