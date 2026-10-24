// A field of a fresh object, read after another object is made: with optimization it is the value
// just stored, not a load. The allocator is declared to LLVM as malloc is: its result is a block no
// other pointer reaches, and it writes nothing the program can read. Declared as writing other
// memory, the second `new` clobbered `v.x`, which was loaded back from the heap; in the ray tracer
// that kept every vector of an expression like `Vector.plus(Vector.times(k, d), p)` alive.
// fresh-objects-folded.cmake reads the optimized IR of `keep`; here the program runs.
class V {
    constructor(public x: number) {}
    static of(x: number) {
        return new V(x);
    }
}

export function keep(a: number, b: number, sink: (v: V) => void) {
    const v = V.of(a);
    const w = V.of(b);
    const r = v.x + w.x;
    sink(v);
    sink(w);
    return r;
}

function main() {
    let n = 0;
    assert(keep(1, 2, (v: V) => { n += v.x; }) == 3 && n == 3, "the fields are the ones stored");
    print("done.");
}
