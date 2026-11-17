// `var` follows the MDN rules: it is scoped to the function it is written in (not the block, not the
// module), it is hoisted, and a redeclaration is the same variable. Each call has its own `var`s.
function rec(n: number): number { var x = n; if (n > 0) rec(n - 1); return x; }
function f() { var x = 10; return x; }
function g() { var x = 1; return x; }
function blockScope(c: boolean) { if (c) { var x = 1; } else { var x = 2; } return x; }
function hoisted() { x = 5; var x: number; return x; }
function redeclared() { var v = 1; var v = 2; var v; return v; }
function param(p: number) { var p = p + 1; return p; }
function twoLoops() {
    var s = 0;
    for (var i = 0; i < 3; i++) { s += i; }
    for (var i = 10; i < 12; i++) { s += i; }
    return s + i;
}
function closures() {
    var fs: (() => number)[] = [];
    for (var i: number = 0; i < 3; i++) { fs.push(() => i); }
    return fs[0]() + fs[1]() + fs[2]();
}
function outlive() {
    var count = 0;
    return () => ++count;
}
function nested() {
    var a = 1;
    function inner() { a = a + 10; return a; }
    const arrow = () => a * 2;
    inner();
    return arrow();
}
function names() {
    var s = "a";
    if (s.length > 0) { var t = s + "b"; s = t + "c"; }
    return s;
}
function forOf() { var t = 0; for (var e of [1, 2, 3]) { t += e; } return t; }
function forIn() { var n = 0; for (var k in [5, 6, 7]) { n++; } return n; }
function* gen() { var a = 1; yield a; var b = a + 1; a = 5; yield b + a; }

function main() {
    assert(rec(3) == 3);
    assert(f() + g() == 11);
    assert(blockScope(true) == 1 && blockScope(false) == 2);
    assert(hoisted() == 5);
    assert(redeclared() == 2);
    assert(param(4) == 5);
    assert(twoLoops() == 36);
    assert(closures() == 9);
    const c = outlive(); c(); c();
    assert(c() == 3);
    assert(nested() == 22);
    assert(names() == "abc");
    assert(forOf() == 6);
    assert(forIn() == 3);
    var s = 0; for (const v of gen()) { s += v; }
    assert(s == 8);
    print("done.");
}
