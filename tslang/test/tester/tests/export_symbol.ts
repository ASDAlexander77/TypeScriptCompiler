// the exporting half of import_symbol.ts: symbols made in another library

export const k = Symbol("k");

export function getK() {
    return k;
}

export function forX() {
    return Symbol.for("x");
}

export function iterator() {
    return Symbol.iterator;
}
