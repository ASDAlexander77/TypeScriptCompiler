// The library side of import_const_locals.ts. Every function reads a `const` local. When a module
// imports this file as source (no shared library beside it), its bodies are compiled only to
// infer types, with no values - and a `const` used to bind its name to that missing value, so the
// importer failed with "can't resolve name". `let` was never affected.

export function plainConst(n: number): number {
    const s = n + 1;
    return s;
}

// No return type: it is inferred from the const, so the importer must infer the same one the
// library does - a const read back as its storage reference instead gave `ref<number>` here.
export function inferredFromConst(n: number) {
    const s = n + 2;
    return s;
}

export function stringConst(n: number): string {
    const s = `v-${n}`;
    return s;
}

export function constInLoop(n: number): number {
    let total = 0;
    for (let j = 0; j < n; j++) {
        const s = j * 2;
        total = total + s;
    }

    return total;
}

export function constInBlock(n: number): number {
    let total = 0;
    if (n > 0) {
        const s = `${n}`;
        total = total + s.length;
    }

    return total;
}

export class WithConst {
    method(n: number) {
        const s = n + 10;
        return s;
    }
}

export function arrayDestructuring(n: number): number {
    const [a, b] = [n, n + 1];
    return a + b;
}

export function objectDestructuring(n: number): number {
    const { x, y } = { x: n, y: 5 };
    return x + y;
}

namespace NS {
    export function namespaceConst(n: number): number {
        const s = `${n}-${n}`;
        return s.length;
    }
}
