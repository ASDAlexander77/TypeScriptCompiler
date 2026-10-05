// reported "Expected 1 arguments, but got 0"
type Sq = { kind: "sq"; side: number };

function f(s: Sq) {
    return s.kind;
}

function main() {
    print(f({ kind: "ci", r: 1 }));
}
