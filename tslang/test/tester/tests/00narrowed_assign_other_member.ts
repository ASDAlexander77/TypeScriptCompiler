// an object literal assigned to a variable narrowed by a discriminant (`s.kind === "sq"`) was typed as the
// narrowed member, so `s = { kind: "ci", r: 1 }` made a `{ kind: "sq", r }` and stored nothing: an `if` kept
// "sq", and a `while` never ended (or crashed the compiler)
type Sq = { kind: "sq"; side: number };
type Ci = { kind: "ci"; r: number };

function inIf(s: Sq | Ci) {
    if (s.kind === "sq") {
        s = { kind: "ci", r: 1 };
    }

    return s.kind;
}

function inWhile(s: Sq | Ci) {
    let n = 0;
    while (s.kind === "sq") {
        n++;
        s = { kind: "ci", r: 2 };
    }

    return n;
}

function sameMember(s: Sq | Ci) {
    if (s.kind === "sq") {
        s = { kind: "sq", side: 9 };
    }

    return s.kind === "sq" ? s.side : -1;
}

function readAfter(s: Sq | Ci) {
    if (s.kind === "sq") {
        s = { kind: "ci", r: 7 };
    }

    return s.kind === "ci" ? s.r : -1;
}

function main() {
    assert(inIf({ kind: "sq", side: 4 }) == "ci", "if: assigned the other member");
    assert(inIf({ kind: "ci", r: 3 }) == "ci", "if: not taken");
    assert(inWhile({ kind: "sq", side: 4 }) == 1, "while: assigned the other member ends the loop");
    assert(inWhile({ kind: "ci", r: 3 }) == 0, "while: not entered");
    assert(sameMember({ kind: "sq", side: 4 }) == 9, "assigned the same member");
    assert(readAfter({ kind: "sq", side: 4 }) == 7, "the other member's field after the if");

    print("done.");
}
