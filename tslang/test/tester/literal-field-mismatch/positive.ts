// literals that are the receiver's type still assign, declare, pass and return
type Sq = { kind: "sq"; side: number };
type Ci = { kind: "ci"; r: number };

function f(s: Sq) {
    return s.side;
}

function g(): Sq {
    return { kind: "sq", side: 3 };
}

function main() {
    let u: Sq | Ci = { kind: "ci", r: 1 };
    assert(u.kind == "ci", "union, declared");
    u = { kind: "sq", side: 2 };
    assert(u.kind == "sq", "union, assigned");

    let s: Sq = { kind: "sq", side: 4 };
    s = { kind: "sq", side: 6 };
    assert(s.side == 6, "assigned");
    assert(f({ kind: "sq", side: 5 }) == 5, "argument");
    assert(g().side == 3, "returned");

    print("done.");
}
