// A library whose exported `A` holds a `Hidden` it does not export: __decls prints `h: Hidden` with
// no declaration of `Hidden` (import_unresolved_iface.ts).
interface Hidden {
    k: number;
}

export interface A {
    h: Hidden;
    n: number;
}

export function mkA(n: number): A {
    return { h: { k: n + 1 }, n: n };
}
