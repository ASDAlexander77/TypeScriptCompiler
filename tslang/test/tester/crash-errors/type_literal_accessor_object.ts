type T = { get foo(): number; set foo(v: number) };
function main() {
    let f: number = 0;
    const t: T = { get foo() { return f; }, set foo(v: number) { f = v * 2; } };
    t.foo = 3;
}
