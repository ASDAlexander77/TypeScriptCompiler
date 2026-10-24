// The library side of @value across a module boundary: a @value class with a static helper and a
// getter, @value interfaces holding it (`Hit` is declared before the `Ray` it holds, so its fields
// are laid out in declaration order on both sides), functions returning them, one returning
// `Hit | null`, and ones reading every field of a value the importer builds. The importer sees only
// these declarations (__decls), which carry `@value`. See import_value_types.ts.
@value
export class V {
    constructor(public readonly x: number) {}

    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }

    get twice() {
        return this.x * 2;
    }
}

@value
export interface R {
    readonly a: V;
    readonly n: number;
}

@value
export interface Hit {
    readonly ray: Ray;
    readonly dist: number;
}

@value
export interface Ray {
    readonly start: V;
    readonly dir: V;
}

export function mkR(n: number): R {
    return { a: new V(n), n: n };
}

export function readR(r: R) {
    return r.a.x * 10 + r.n;
}

export function mkHit(n: number): Hit {
    return { ray: { start: new V(n), dir: new V(n + 1) }, dist: n * 2 };
}

export function readHit(h: Hit) {
    return h.ray.start.x + h.ray.dir.x * 10 + h.dist * 100;
}

export function find(n: number): Hit | null {
    if (n < 0) return null;
    return mkHit(n);
}
