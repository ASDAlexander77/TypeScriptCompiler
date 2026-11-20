// A key of a type (`P extends keyof T`) keeps the literal it is given, as in TypeScript: `T[P]` is that field's type
// (widened to `string` it was "indexing a type by 'string' is not supported"), and `obj[key]` with `key: P` reads the
// field the literal names ("indexer is not declared").
interface O {
    m?: number[];
    n: string;
}

function get<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

function orMake<T, P extends keyof T>(obj: T, prop: P, factory: () => T[P]): T[P] {
    return factory();
}

function main() {
    const o: O = { n: "x" };
    assert(get(o, "n") == "x");
    const r = orMake(o, "m", (): number[] => [1, 2]);
    assert(r.length == 2);
    print("done.");
}
