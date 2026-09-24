// An optional call whose result is used, inside a larger expression, at module level: the
// failed call must take its half-built `if` with it, not leave it in the global's initializer.
let f: ((a: number, b: number) => number) | undefined = (a: number, b: number) => a + b;
const r = `${f?.(1) ?? 0}`;

function main() {
    print(r);
}
