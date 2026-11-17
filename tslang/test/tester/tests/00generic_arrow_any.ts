// A generic function value nothing gives types to holds one function, its untyped parameters `any`,
// as TypeScript types them: in a `let` or `var` (module or local), returned from a function with no
// declared return type, or as an object literal field. Kept generic, a module variable named a function
// that was never made ("'llvm.mlir.addressof' op must reference a global", TSLANG 106) and a local crashed.
let foo = (bar => bar);
var g = <U>(u: U) => u;
function local() { let f = (x => x); var h = <U>(u?: U) => { }; h(1); return f(5); }
function returned() { return r => r; }
function adder(n: number) { return (m => m + n); }
function nested() { var result; if (true) { var a2 = x => x; result = a2(4); } return result; }
function twice<T>(x: T) { return x; }
var o = {
    f: a => 1,
    g: (a => a),
    twice
};

function main() {
    assert(foo(3) == 3);
    assert(g("s") == "s");
    assert(local() == 5);
    assert(returned()(7) == 7);
    assert(adder(2)(3) == 5);
    assert(nested() == 4);
    assert(o.f(0) == 1 && o.g(5) == 5 && o.twice("t") == "t");
    print("done.");
}
