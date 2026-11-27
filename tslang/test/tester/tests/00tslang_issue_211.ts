// TSLANG 211, every model: leftovers of 193, 194, 205 and old "still open" notes (137, own conversions).

// 193: a discriminant read from an exported module `const` (stored as a `let`, user ruling) kept its string literal
// type only as a string, and a literal told apart by it matched no member ("can't be stored")
type KindA = { kind: "a"; method(x: string): number };
type KindB = { kind: "b"; method(x: number): number };
export const KA = "a";

function run(o: KindA | KindB) {
    return o.kind == "a" ? o.method("xyz") : o.method(5);
}

// 194: `"msg" in v` of a union of object shapes (it was "only a union of interfaces is"), and `v.msg` read from the
// member the value holds (it was read as the first member that has it, past the end of a smaller one)
type Validated = { valid: boolean } | { valid: boolean; msg: string };

function makeValidated(b: boolean): Validated {
    if (b) {
        return { valid: true };
    }

    return { valid: false, msg: "m" + 1 };
}

function msgOf(v: Validated) {
    if ("msg" in v) {
        return v.msg;
    }

    return "none";
}

// 205: `X & any` is `any` (it was `never`)
type WithAny = { x: string } & any;

function passAny(w: WithAny) {
    return w;
}

// own: a variable holding a string converted to another object type (`n` an s32 for a number), stored, read by a
// callee and kept by one
function nameLength(p: { n: number; name: string }) {
    return p.name.length + p.n;
}

class Keeper {
    all: { n: number; name: string }[] = [];

    keep(p: { n: number; name: string }) {
        this.all.push(p);
    }
}

function converted(i: number) {
    const u = { n: i, name: "m" + i };
    const p: { n: number; name: string } = u;
    return p;
}

class ObjA {
    constructor(public k: number) {}

    n(): number {
        return 1 + this.k;
    }
}

class ObjB {
    constructor(public s: string) {}

    n(): number {
        return this.s.length;
    }
}

function main() {
    assert(run({ kind: KA, method(a) { return a.length; } }) == 3, "exported const discriminant");

    assert(msgOf(makeValidated(false)) == "m1" && msgOf(makeValidated(true)) == "none", "in of shapes");
    const withMsg: Validated = makeValidated(false);
    assert(withMsg.valid == false, "a field every member has");

    const anyValue: any = { x: "s" };
    passAny(anyValue);

    const k = new Keeper();
    let total: number = 0;
    for (let i = 0; i < 3; i++) {
        const u = { n: i, name: "x" + i };
        const p: { n: number; name: string } = u;
        total += p.name.length;
        total += nameLength(u);
        k.keep(u);
        assert(u.name == "x" + i, "source kept its string");
    }

    assert(total == 15 && k.all.length == 3 && k.all[2].name == "x2" && converted(4).name == "m4", "conversion");

    // 137: for...of straight over a literal of objects (under -mm=own "moved only into a local")
    let s: number = 0;
    for (let i = 0; i < 3; i++) {
        for (const o of [new ObjA(i), new ObjB("b" + i)]) s += o.n();
    }

    assert(s == 12, "for of literal");

    print("done.");
}
