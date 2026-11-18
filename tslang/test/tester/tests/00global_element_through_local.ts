// An argument read out of a global's array or object through a local that names it (`const a = GA;
// f(a[0])`) is held across a call that may give that place a new value, as one read through the
// global itself is (-mm=rc). The callee's `GA[0] = ...` freed the object it still used through its
// parameter, and a write through the parameter corrupted the heap (fuzz 946037, Linux).

class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class N {
    v: V;
    tag: string;
    constructor(v: V, tag: string) {
        this.v = v;
        this.tag = tag;
    }
}

function mk(i: number): N {
    return new N(new V(i), "n" + i);
}

const GA: N[] = [mk(2)];
let GO: N = mk(3);

function replaceElement(p: N, i: number): number {
    GA[0] = mk(i + 100);
    p.v = new V(i);
    return p.v.x + p.tag.length;
}

function replaceField(p: V, i: number): number {
    GO.v = new V(i + 100);
    const w = new V(i + 200);
    return p.x + w.x;
}

function main() {
    for (let i = 0; i < 1000; i++) {
        const a: N[] = GA;
        const before = a[0].tag.length;
        assert(replaceElement(a[0], i) == i + before, "an element through a local");
        assert(GA[0].v.x == i + 100, "the global's new element");

        const o: N = GO;
        const x = o.v.x;
        assert(replaceField(o.v, i) == x + i + 200, "a field through a local");
    }

    print("done.");
}
