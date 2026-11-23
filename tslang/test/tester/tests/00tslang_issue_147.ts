// TSLANG_ISSUES 147: an identifier starting with a braced Unicode escape (`\u{61}`) failed with
// "Hexadecimal digit expected": the rest of the name was scanned before the escape, and took its hex
// digits. Plain characters between two braced escapes were dropped too.

function main() {
    const \u{61} = 1;
    const \u{62}\u{63} = 2;
    const \u{64}x\u{65} = 3;
    const f\u{6F}o = 5;
    assert(a == 1, "start");
    assert(bc == 2, "two escapes");
    assert(dxe == 3, "a character between two escapes");
    assert(foo == 5, "inside a name");
    print("done.");
}
