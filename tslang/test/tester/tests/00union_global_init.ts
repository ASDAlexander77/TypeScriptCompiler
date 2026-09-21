// Module-level and class-static unions initialized with a constant, holding the largest member
// and a smaller one. A tagged union's value goes in through memory (an alloca and a copy), which
// a global's initializer cannot hold, so these globals are set up by a global constructor.
//
// The end of the file checks that a union's unused storage bytes are zero: two equal numbers
// boxed into `any` from `number | P` compare equal. The comparison looks at the whole boxed
// storage, so bytes the number does not cover must not be left over from an earlier P.

let gLetBig: number | string = 2.5;
let gLetSmall: number | string = "x";
const gConstBig: number | string = 4.5;
const gConstSmall: number | string = "y";
let gFlag: s32 | boolean = true;

type Three = { flag: boolean; a: s32; b: s32 };
type Big = { flag: boolean; v: number };
const gThree: Three | Big = { flag: true, a: 3, b: 4 };

class C {
    static sBig: number | string = 1.5;
    static sSmall: number | string = "s";
}

function show(u: number | string): string {
    if (typeof u == "number") return `n${u}`;
    return `s${u}`;
}

type P = { a: number; b: number };
function mkP(): number | P { return { a: 7.25, b: 9.75 }; }
function mkN(v: number): number | P { return v; }
function boxP(): any { const u = mkP(); const x: any = u; return x; }
function boxN(v: number): any { const u = mkN(v); const x: any = u; return x; }

function main() {
    assert(show(gLetBig) == "n2.5", "let, largest member");
    assert(show(gLetSmall) == "sx", "let, smaller member");
    assert(show(gConstBig) == "n4.5", "const, largest member");
    assert(show(gConstSmall) == "sy", "const, smaller member");
    assert(show(C.sBig) == "n1.5", "static, largest member");
    assert(show(C.sSmall) == "ss", "static, smaller member");
    assert(typeof gFlag == "boolean", "s32 | boolean tag");

    const three = <Three><any>gThree;
    assert(three.flag, "global union three.flag");
    assert(three.a == 3, "global union three.a");
    assert(three.b == 4, "global union three.b");

    gLetSmall = 8.5;
    assert(show(gLetSmall) == "n8.5", "let, reassigned");

    const p = boxP();
    const x = boxN(1.5);
    const y = boxN(1.5);
    assert(x === y, "equal numbers boxed from unions");

    print("done.");
}
