// `cond ? new C() : null` hands the new object to whoever takes the result (#465). Under rc the
// branch released it as a discarded temporary before the merge, so the result pointed at freed
// memory. Each result is read after other allocations have reused whatever was freed.
class Node {
    v = 5;
    name = "";
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

function nodeOrNull(set: boolean): Node | null {
    return set ? new Node() : null;
}

function named(set: boolean, name: string): Node | undefined {
    const n = new Node();
    n.name = name;
    return set ? n : undefined;
}

function main() {
    const set = true;
    const n: Node | null = set ? new Node() : null;
    assert(churn() == 1000);
    assert(n !== null && n.v == 5, "a local from a ternary keeps its node");

    const m = nodeOrNull(true);
    assert(churn() == 1000);
    assert(m !== null && m.v == 5, "a returned ternary keeps its node");

    assert(nodeOrNull(false) === null, "the null side is null");

    const k = named(true, "kept" + 1);
    assert(churn() == 1000);
    assert(k !== undefined && k.name == "kept1", "an optional ternary keeps its node");

    for (let i = 0; i < 1000; i++) {
        const t = i % 2 == 0 ? new Node() : null;
        if (t) t.v = i;
    }

    print("done.");
}
