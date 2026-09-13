// The library side of import_const_functions.ts. An exported module-level `const` holding a
// function is a variable, as an exported `let` is: importers read the function pointer out of it.
// Importers used to look for such a variable while the library had turned the const into a
// function and erased it - an undefined symbol when linked, a call through the function's code
// bytes under the JIT.

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

export function namedFunction(n: number) {
    return n * 7;
}

// holds a function made under another name
export const alias = namedFunction;

export let letArrow = (n: number) => n * 8;

const notExported = (n: number) => n + 100;

export function usesNotExported(n: number) {
    return notExported(n);
}

// a generator's wrapper is not named after the const, so the const must keep its global
const localGenerator = function* () {
    yield 3;
    yield 4;
};

export function usesLocalGenerator() {
    let sum = 0;
    for (const v of localGenerator()) {
        sum = sum + v;
    }

    return sum;
}

namespace NS {
    export const inNamespace = (n: number) => n * 6;
}
