// -mm=own: a function's `var` is a local of that function, hoisted, one variable per call; a value
// it holds is released when the function returns or when the var is given a new one.
class Item { constructor(public n: number) {} }
function keep() {
    var last = new Item(0);
    for (var i = 1; i < 4; i++) { last = new Item(i); }
    return last;
}
function names() {
    var s = "a";
    if (s.length > 0) { var t = s + "b"; s = t + "c"; }
    return s;
}
function rec(n: number): number { var x = n; if (n > 0) rec(n - 1); return x; }
function forOf() { var t = 0; for (var e of [1, 2, 3]) { t += e; } return t; }
function* gen() { var a = 1; yield a; var b = a + 1; a = 5; yield b + a; }
function main() {
    for (let k = 0; k < 3; k++) {
        assert(keep().n == 3);
        assert(names() == "abc");
        assert(forOf() == 6);
    }
    assert(rec(3) == 3);
    var s = 0; for (const v of gen()) { s += v; }
    assert(s == 8);
    print("done.");
}
