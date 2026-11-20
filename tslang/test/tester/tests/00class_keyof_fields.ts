// A class read by a key naming fields only, under every model -mm=own included: the member's own type, no any
class C {
    a = 1;
    b = 2;
    s = "x";
}

function pick(i: number): "a" | "b" {
    return i == 0 ? "a" : "b";
}

function pickAny(i: number): "a" | "s" {
    return i == 0 ? "a" : "s";
}

function main() {
    const c = new C();
    assert(c[pick(1)] == 2);
    c[pick(0)] = 5;
    assert(c.a == 5);
    const v = c[pickAny(1)];
    assert(typeof v == "string" && <string>v == "x");
    print("done.");
}
