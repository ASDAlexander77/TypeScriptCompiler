// -mm=own rejects: `n` moves into what `set ? n : m` selects, and is read after it.
class Node {
    name = "";
}

function pick(set: boolean, i: number) {
    const n = new Node();
    n.name = "n" + i;
    const m = new Node();
    const r = set ? n : m;
    print(n.name);
    return r;
}

function main() {
    print(pick(true, 1).name);
}
