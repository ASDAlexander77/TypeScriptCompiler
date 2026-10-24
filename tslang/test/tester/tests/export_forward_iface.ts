// The library side of a forward reference across a module boundary: `A` names `B`, declared after
// it, and __decls prints them in that order. The importer reads each declaration again once the
// others are in, as a module reads its own (import_forward_iface.ts).
export interface A {
    b: B;
    n: number;
}

export interface B {
    k: number;
}

export function mkA(n: number): A {
    return { b: { k: n * 10 }, n: n };
}
