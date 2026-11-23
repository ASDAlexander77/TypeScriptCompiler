// TSLANG 170: under -mm=rc an `object` is an `any` box (USER RULING 2026-11-23), so a class object whose last
// reference is an `object` is freed with its fields (the loop below, flat under rc). gc and none box it too (171), so an `any`
// taken from an `object` is what it holds there as well. Not under -mm=own, which refuses `const a: any = o` of an `object`.

class L {
    s: string;
    constructor(i: number) {
        this.s = "s" + i;
    }
}

class H {
    o: object;
    constructor(o: object) {
        this.o = o;
    }
}

function keep(o: object) {
    return o;
}

function main() {
    const c = new L(1);
    const o = <object>c;
    assert(o === c);
    assert(c === o);
    const o2: object = c;
    assert(o === o2);
    const other: object = new L(2);
    assert(o !== other);
    assert(keep(c) === c);
    assert(typeof o == "object");

    let n: object | null = null;
    assert(n === null);
    assert(!n);

    const a: any = o;
    const u: unknown = o;
    assert(a === u);
    assert(a === o);
    assert(u === o);
    assert((<L>a).s == "s1");

    let t = 0;
    for (let i = 0; i < 1000; i++) {
        const k = <object>new L(i);
        const arr: object[] = [new L(i), new L(i + 1)];
        const h = new H(arr[0]);
        const lit = <object>{ a: "a" + i, b: "b" + i };
        const w: object | string = i % 2 == 0 ? <object>new L(i) : "x" + i;
        if (k && h.o === arr[0] && lit && w) t++;
    }
    assert(t == 1000);

    print("done.");
}