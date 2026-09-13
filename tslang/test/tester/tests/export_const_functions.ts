// The library side of import_const_functions.ts. A module-level `const` initialized with an arrow
// function or a function expression compiles to a function of that name, with no variable. Its
// importers used to look for a variable holding a pointer instead: an undefined symbol when linked,
// a call through the function's code bytes under the JIT.

export const arrowBlock = (n: number) => {
    let s = n * 3;
    return s;
};

export const arrowExpression = (n: number) => n * 4;

export const functionExpression = function (n: number) {
    return n * 5;
};

export const arrowWithConstLocal = (n: number) => {
    const s = `v-${n}`;
    return s;
};

const notExported = (n: number) => n + 100;

export function usesNotExported(n: number) {
    return notExported(n);
}

namespace NS {
    export const inNamespace = (n: number) => n * 6;
}
