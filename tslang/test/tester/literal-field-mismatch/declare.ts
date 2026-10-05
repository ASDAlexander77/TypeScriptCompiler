// compiled with no error, and `t` held garbage
type Sq = { kind: "sq"; side: number };

function main() {
    let t: Sq = { kind: "ci", r: 1 };
    print(t.kind);
}
