// a character outside the BMP in a string literal: the parser holds it as a UTF-16 surrogate pair,
// and on Linux (32-bit wchar_t) the literal came out as "" - strings are UTF-8, so it is 4 bytes
const globalSmile = "😀";

function bytes(s: string) {
    return s.length;
}

function main() {
    const smile = "😀";
    assert(smile.length == 4, "a literal outside the BMP is four bytes");
    assert(bytes(globalSmile) == 4, "a global literal outside the BMP");
    assert(smile == "\u{1F600}", "the same as a code point escape");
    assert(smile == "😀", "the same as a surrogate pair escape");
    assert(`<${smile}>`.length == 6, "in a template");
    assert(`😀`.length == 4, "a template literal outside the BMP");
    assert("a😀b".length == 6, "between other characters");
    assert("é€😀".length == 9, "two, three and four bytes");

    print(smile, "done.");
}
