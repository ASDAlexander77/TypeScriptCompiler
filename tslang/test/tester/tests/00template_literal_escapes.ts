// Escape sequences around a substitution in a template literal stayed as written: `${n}\n${n}`
// held the four characters 1, \, n, 1. A template without substitutions and a string literal
// were fine.

type Name = "you";
// a template literal type: "hi\tyou" did not match it, as the type kept a backslash and a t
type Greeting = `hi\t${Name}`;

function tag(strings: string[], a: number, b: number): string {
    return strings[0] + "|" + strings[1] + "|" + strings[2];
}

function main() {
    const n = 1;

    const s = `${n}\n${n}`;
    assert(s.length == 3, "middle escape");
    assert(s == "1\n1", "middle text");

    const head = `\t${n}`;
    assert(head.length == 2, "head escape");
    assert(head == "\t1", "head text");

    const tail = `${n}\\`;
    assert(tail.length == 2, "tail escape");
    assert(tail == "1\\", "tail text");

    const several = `a\x41${n}Bb`;
    assert(several == "aA1Bb", "hex and unicode escapes");

    const quote = `\`${n}\``;
    assert(quote == "`1`", "escaped backticks");

    const plain = `a\nb`;
    assert(plain.length == 3, "no substitution");

    const tagged = tag`x\ty${n}\n${2}z`;
    assert(tagged == "x\ty|\n|z", "tagged template strings are cooked");

    const greeting: Greeting = "hi\tyou";
    assert(greeting.length == 6, "template literal type");

    print("done.");
}
