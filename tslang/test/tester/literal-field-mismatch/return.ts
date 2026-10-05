// a returned literal of the other member
type Sq = { kind: "sq"; side: number };

function g(): Sq {
    return { kind: "ci", r: 1 };
}

function main() {
    print(g().kind);
}
