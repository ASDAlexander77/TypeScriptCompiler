// `@owned` on a declaration: the function is TypeScript defined elsewhere under the same memory model,
// so under rc and own its result is a value the caller owns (DECLARED_OWNED_RESULT_ATTR_NAME). Said on a
// `declare function`, a `declare class` (each method, and `..new`) or a `declare namespace` (each
// function in it). A method that says it lends its result (`@lends`) is own's to go by: under own its
// result is a borrow, given back by no one. Read by owned-declarations.cmake, as it is and with every
// `@owned` taken out, under rc and under own.

@owned
declare function lib_text(n: number): string;

@owned
declare class LibBox {
    constructor(n: number);
    describe(): string;
    @lends("this")
    label(): string;
}

@owned
declare namespace LibNS {
    function name(this: number): string;
}

class P {
    constructor(public x: number) {}
}

function make(n: number): P {
    return new P(n);
}

// a call through a parameter names no callee: its result is consumed only when no function here may
// return a heap value without retaining it - and the declarations above are the only ones that could
// (`label` among them under own, so own reads `viaValue` as unconsumed either way)
function viaValue(g: (n: number) => P, n: number) {
    const p = g(n);
    return p.x;
}

// a declared function's result, read once and dropped: the caller's to give back
function direct(n: number) {
    return lib_text(n).length;
}

// a borrow of `b`: never the caller's under own
function lent(b: LibBox) {
    return b.label().length;
}

function main() {
    print(viaValue(make, 1));
}
