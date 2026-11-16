// -mm=own: an object literal assigned to a global (`o = { s: mk(i), n: i }`) moves into it, with
// what its fields were given; the global's old record is released where it gets the new one.
class Node {
    v = 0;
    name = "";
}

function mk(name: string) {
    const n = new Node();
    n.name = name;
    return n;
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 100; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

let o = { s: mk("made" + 1), n: 7 };

function main() {
    for (let i = 0; i < 5000; i++) {
        const k = i % 10;
        o = { s: mk("next" + k), n: k };
        assert(churn() == 100);
        assert(o.s.name == "next" + k && o.n == k, "a reassigned global literal keeps the new object");
    }

    print("done.");
}
