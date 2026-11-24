// -mm=own --own-fallback=count with the default library (own-defaultlib-facts.cmake, EXTRA): a node own refuses
// to give two owners (`link` stores it in two holders) is counted, and the counted node is then read through the
// library's array helpers (`map`, `find`, `some`, `indexOf`; `filter` of the values it maps to) and its text kept in a Map and a Set (own's
// Map and Set hold values, strings, `any` and Shared<T>), beside its holders. Each owner gives its share back; the
// program prints what gc prints.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function link(h: Holder, g: Holder, n: Node) {
    h.node = n;
    g.node = n;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 200; i++) {
        const h = new Holder();
        const g = new Holder();
        link(h, g, new Node(i, "a" + i));

        const byName = new Map<string, number>();
        byName.set(h.node!.s, h.node!.v);
        const seen = new Set<string>();
        seen.add(g.node!.s);
        const all: Node[] = [new Node(i + 1, "b" + i), new Node(i + 2, "c" + i)];
        const values = all.map((n) => n.v);
        const even = values.filter((v) => v % 2 == 0);
        const found = all.find((n) => n.v == i + 2);
        const any = all.some((n) => n.s == "b" + i);

        sum += byName.get("a" + i)! + (seen.has(h.node!.s) ? 1 : 0) + even.length + values[0] + (found ? found.v : 0) +
               (any ? 1 : 0) + all.indexOf(all[1]);
        if (h.node!.s != g.node!.s) {
            print("bad");
        }
    }

    print(sum);
    assert(sum == 61100, "sum");
    print("done.");
}
