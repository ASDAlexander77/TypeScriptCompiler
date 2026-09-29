// -mm=own across a module boundary, the library side: classes an importer builds and uses. See
// import_own_class.ts.
export class C {
    v: number[] = [];
    constructor(public x: number) {}

    twice() {
        return this.x * 2;
    }
}

export class H {
    c: C = new C(0);

    get cx() {
        return this.c.x;
    }

    add(n: number) {
        this.c.v.push(n);
    }
}

export function sumOf(h: H) {
    let s = 0;
    for (let i = 0; i < h.c.v.length; i++) {
        s += h.c.v[i];
    }

    return s;
}
