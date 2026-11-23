// Shapes told apart by their first field (`kind: "sq"` / `kind: "ci"`) are stored without a tag: the field is the
// tag (user ruling 2026-11-23). Read, switched on, held in a field, widened with null, boxed; rc and own release
// what the member the field names holds.
type Sq = { kind: "sq"; size: number; label: string };
type Ci = { kind: "ci"; r: number; name: string; tags: string[] };
type Shape = Sq | Ci;

function make(i: number): Shape {
    if (i % 2 == 0) return { kind: "sq", size: i, label: "s" + i };
    return { kind: "ci", r: i, name: "c" + i, tags: ["t" + i, "u"] };
}

function describe(s: Shape) {
    switch (s.kind) {
        case "sq": return s.label + ":" + s.size;
        case "ci": return s.name + ":" + s.tags.length;
    }
}

class Holder { shape: Shape = make(0); }

function main() {
    let total = 0;
    for (let i = 0; i < 6; i++) {
        const s = make(i);
        const d = describe(s);
        total += d.length;
        const h = new Holder();
        h.shape = s;
        assert(describe(h.shape) == d, "held");
        const n: Shape | null = i % 3 == 0 ? null : s;
        if (n !== null) assert(describe(n) == d, "widened with null");
        const a: any = make(i);
        assert(typeof a == "tuple", "boxed");
    }
    assert(describe(make(4)) == "s4:4", "sq");
    assert(describe(make(5)) == "c5:2", "ci");
    print("total", total);
    print("done.");
}
