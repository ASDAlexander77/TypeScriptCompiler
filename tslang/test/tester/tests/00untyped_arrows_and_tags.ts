// An arrow with untyped parameters that nothing gives types to is instantiated with `any` where it is an
// array element (`[x => x + 1]`) or given as `any` (`g(r => r * 2)` with `g(a: any)`), and the boxed
// one can be called through the `any`; a generic tag (`id\`...\``) is called as the call `id([...])`.
function callAny(a: any) { return a(5); }

function id<T>(x: T) { return x; }

function tag(parts: string[], ...vals: number[]) { return parts.length * 100 + vals[0] * 10 + vals[1]; }

function main() {
    const fs = [x => x + 1];
    assert(fs[0](2) == 3);

    assert(callAny(r => r * 2) == 10);

    const parts = id`hello world`;
    assert(parts.length == 1 && parts[0] == "hello world");

    const n = 5;
    assert(tag`a${n}b${n + 1}c` == 356);

    print("done.");
}
