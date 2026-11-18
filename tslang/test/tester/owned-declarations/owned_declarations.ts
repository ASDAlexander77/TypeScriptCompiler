// `@owned` on a declaration: the function is TypeScript defined elsewhere under the same memory model,
// so under rc its result is a reference the caller owns (DECLARED_OWNED_RESULT_ATTR_NAME). Said on a
// `declare function`, a `declare class` (each method, and `..new`) or a `declare namespace` (each
// function in it). Read by owned-declarations.cmake, as it is and with every `@owned` taken out.

@owned
declare function lib_text(n: number): string;

@owned
declare class LibBox {
    constructor(n: number);
    describe(): string;
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
// return a heap value without retaining it - and the three declarations above are the only ones that
// could
function viaValue(g: (n: number) => P, n: number) {
    const p = g(n);
    return p.x;
}

function main() {
    print(viaValue(make, 1));
}
