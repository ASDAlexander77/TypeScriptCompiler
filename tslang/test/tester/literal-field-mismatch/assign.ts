// #513: compiled with no error, and `s` kept "sq"
type Sq = { kind: "sq"; side: number };

function main() {
    let s: Sq = { kind: "sq", side: 4 };
    s = { kind: "ci", r: 1 };
    print(s.kind, s.side);
}
