type T = { get foo(): number; set foo(v: number) };
class C { _f = 0; get foo(): number { return this._f; } set foo(v: number) { this._f = v; } }
function main() { const t: T = new C(); t.foo = 3; }
