// An object literal that owns what it holds (a fresh string, a new object, an array literal) given
// as an element of an array literal, or pushed, moves into the array as the array's element: the
// array's release frees what the literal held, and nothing else releases it. Before, only an owning
// local, the result or a box could take such a literal ("is moved only into a local or returned as
// it is"). Each step churns, so a value freed too early reads garbage; every model runs it.
class C { constructor(public x: number) {} }
function label(i: number) { return "s" + i; }
function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}
function main() {
    let ok = 0;
    for (let i = 0; i < 1000; i++) {
        const patterns = [
            { name: label(i), order: [1, 2, 3], c: new C(i) },
            { name: "b", order: [3, 1, 2], c: new C(2) },
        ];
        churn();
        const more: { name: string, order: int[] }[] = [];
        for (let j = 0; j < 3; j++) more.push({ name: label(j), order: [j] });
        churn();
        if (patterns[0].name == label(i) && patterns[0].order[1] == 2 && patterns[0].c.x == i && patterns[1].c.x == 2) ok++;
        if (more.length == 3 && more[2].name == "s2" && more[2].order[0] == 2) ok++;
    }
    assert(ok == 2000, "ok");
    print("done.");
}
