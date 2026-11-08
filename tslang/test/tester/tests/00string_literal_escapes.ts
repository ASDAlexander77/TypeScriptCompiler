// A string literal holds what its escapes say, and .length counts its UTF-8 bytes.
// TSLANG_ISSUES 34: a literal starting with U+FEFF lost it - the conversion to UTF-8 took it for a
// byte order mark. TSLANG_ISSUES 35: "\\" is one backslash, so "\\n" is the two characters \ and n.

function main() {
    // 34: U+FEFF is three bytes wherever it is, written as an escape or as itself
    assert("\uFEFF".length == 3, "a literal that is U+FEFF");
    assert("\uFEFFx".length == 4, "a literal starting with U+FEFF");
    assert("a\uFEFFb".length == 5, "U+FEFF in the middle");
    assert("﻿x".length == 4, "a literal starting with U+FEFF typed as is");
    assert("\uFEFFx" == "\u{FEFF}x", "the same as the code point escape");
    // U+FFFE is no byte-swapped order mark either: the rest is not swapped
    assert("\uFFFEab".length == 5, "a literal starting with U+FFFE");
    assert("\uFFFEab" != "\uFFFE\u6100\u6200", "keeps its other characters as they are");
    assert("\u00E9".length == 2 && "\u200B".length == 3 && "\uFFFD".length == 3, "other escapes");

    // 35: an escaped backslash is one backslash, and what follows it is itself
    assert("\\n".length == 2, "\\n");
    assert("\\u0041".length == 6, "\\u0041");
    assert("\\x41".length == 4, "\\x41");
    assert("\\d".length == 2, "\\d");
    assert("\\\\".length == 2, "\\\\");
    assert("a\\b".length == 3, "a\\b");
    const s = "\\n";
    assert(s == "\x5Cn" && s == "\u005Cn" && s != "\n", "a backslash, then n");

    print("done.");
}
