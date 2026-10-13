// regression test: a string a C function returns is copied into a string of the program's own.
//
// bug found and fixed: the string was C's memory, kept as it was. A buffer C reuses changed under
// the program (`strerror(2)` read the text of the next call), and under -mm=rc the count was kept
// in the 8 bytes before the pointer - someone else's memory: storing `strchr(base, 'k')` rewrote
// two characters of `base`. Null stays null.

declare function strchr(s: string, c: s32): string;
declare function strerror(errnum: s32): string;
declare function getenv(name: string): string;

let kept = "";

function tail(s: string): string {
    return strchr(s, 107);   // 'k'
}

function main() {
    let n = 1;
    const base = "abcdefghijklmnop" + n;

    // a pointer into base, stored and copied around: base is untouched
    kept = strchr(base, 107);
    let other = kept;
    for (let i = 0; i < 3; i++) {
        let t = tail(base);
        assert(t == "klmnop1");
    }

    assert(base == "abcdefghijklmnop1");
    assert(kept == "klmnop1");
    assert(other == "klmnop1");

    // two results of one C function are two strings
    const a = strerror(2);
    const b = strerror(13);
    assert(a != b);

    // a copy of nothing is nothing
    const missing = getenv("TSLANG_C_STRING_COPY_TEST_UNSET_VARIABLE");
    assert(missing == null);

    print("done.");
}
