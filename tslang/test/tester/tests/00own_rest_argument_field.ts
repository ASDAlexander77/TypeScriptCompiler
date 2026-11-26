// A field of a record returned by a call, passed straight to a rest parameter (`console.log(mk().value)`,
// `const v = mk().value; console.log(v)` in a function, `const [a] = set; console.log(a)`): the rest
// array borrows the field for the call (classifyArgumentArrays), and the record is given back after it.
// -mm=own refused it ("a value takes a second reference"): the field's one use, the borrowing array,
// was not counted as a read of the record. Runtime strings, not literals: a literal is immortal and
// hides a wrong count.
function mk(i: number) {
    return { value: "x" + i, done: false };
}

function* strings(k: number) {
    yield "g" + k;
    yield "h" + k;
}

function count(...items: string[]): number {
    let n: number = 0;
    for (const s of items) {
        n += s.length;
    }

    return n;
}

function named(i: number): number {
    const v = mk(i).value;
    return count(v);
}

function direct(i: number): number {
    return count(mk(i).value, mk(i + 1).value);
}

function destructured(i: number): number {
    const { value } = mk(i);
    return count(value);
}

function held(i: number): number {
    const r = mk(i);
    return count(r.value);
}

function fromNext(i: number): number {
    const it = strings(i);
    const w = it.next().value;
    return count(w ?? "");
}

function fromPattern(i: number): number {
    const [a, b] = strings(i);
    return count(a ?? "", b ?? "");
}

function main() {
    assert(named(5) == 2, "named");
    assert(direct(5) == 4, "direct");
    assert(destructured(5) == 2, "destructured");
    assert(held(5) == 2, "held");
    assert(fromNext(5) == 2, "next().value");
    assert(fromPattern(5) == 4, "array pattern of an iterable");

    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        n += named(i) + direct(i) + destructured(i) + held(i) + fromNext(i) + fromPattern(i);
    }

    assert(n > 0, "many times");

    print("done.");
}
