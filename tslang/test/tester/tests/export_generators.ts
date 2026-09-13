// The library side of import_generators.ts. A generator returns its object by reference; the
// declaration text importers read (__decls) used to print that object as a plain `{...}`, which
// reads back as a value, so a -shared importer got garbage out of next().

export function* functionGenerator() {
    yield 1;
    yield 2;
}

export const constGenerator = function* () {
    yield 3;
    yield 4;
};

export class WithGenerator {
    *items() {
        yield 5;
        yield 6;
    }
}

namespace NS {
    export function* inNamespace() {
        yield 7;
        yield 8;
    }
}
