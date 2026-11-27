// TSLANG 213: a class given to a construct signature of another result type (a type literal's, an interface's), an
// interface method an object literal's function backs, and `any` compared with undefined.

interface Pt {
    x: number;
}

interface PtCtor {
    new (x: number): Pt;
}

class P implements Pt {
    s: string;
    constructor(public x: number) {
        this.s = "s" + x;
    }
}

// a type literal's construct signature returning an interface: the function made for the class returns the interface
// (it returned the class, read by the caller as the interface: garbage, or a crash)
function makeByLiteral(c: { new (x: number): Pt }, i: number): Pt {
    return new c(i);
}

function makeByInterface(c: PtCtor, i: number): Pt {
    return new c(i);
}

interface Mk {
    make(x: number): P;
}

function isUndefined(v: any) {
    return v === undefined;
}

function isNullish(v: any) {
    return v == undefined;
}

function main() {
    let total = 0;
    for (let i = 0; i < 100; i++) {
        total += makeByLiteral(P, i).x + makeByInterface(P, i).x;
    }

    assert(total == 9900);
    assert(makeByLiteral(P, 4).x == 4);

    const m: Mk = { make: (x: number) => new P(x) };
    let made = 0;
    for (let i = 0; i < 10; i++) {
        made += m.make(i).s.length;
    }

    assert(made == 20);

    const values: any[] = [undefined, null, 0, "", "x", false];
    let flags = "";
    for (const v of values) {
        flags += (isUndefined(v) ? "1" : "0") + (isNullish(v) ? "1" : "0") + " ";
    }

    assert(flags == "11 01 00 00 00 00 ");

    // no initializer: undefined in every model (00let_any_starts_undefined.ts)
    let u: any;
    assert(u === undefined);
    assert(!(u === null));

    print("done.");
}
