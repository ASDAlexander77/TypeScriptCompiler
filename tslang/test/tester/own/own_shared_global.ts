// Shared<T> (spec 23.1) in module-level globals, bare and `Shared<T> | null`, each initialised by
// `new Shared(...)` at module level: each global is born holding one count. Each test copies a
// global, drops the global's own handle, reuses the memory, then reads through the copy; and the
// global that keeps its handle is read after its copy is dropped. Counts are checked where there
// is one (rc, own); gc and none report -1.
class Node {
    v = 0;
    tag = "";
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.tag = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

function countIs(s: Shared<Node>, expected: number) {
    const c = Shared.count(s);
    return c < 0 || c == expected;
}

function mk(tag: string, v: number) {
    const n = new Node();
    n.tag = tag;
    n.v = v;
    return n;
}

const root = new Shared(mk("root", 1));
let bare = new Shared(mk("bare", 2));
let maybe: Shared<Node> | null = new Shared(mk("maybe", 3));

// a const global keeps its handle: a copy is dropped, and the global still reads its block
function constGlobal() {
    assert(countIs(root, 1), "a const global is born with one count");
    let x = root;
    assert(countIs(root, 2), "copied");
    x = new Shared(mk("other", 9));
    assert(churn() == 1000);
    assert(countIs(root, 1), "the copy dropped");
    assert(root.value.v == 1 && root.value.tag == "root", "the const global's block is alive");
    assert(countIs(x, 1));
}

// a let global's handle replaced: the copy is the only handle left
function bareGlobal() {
    assert(countIs(bare, 1), "a let global is born with one count");
    const h = bare;
    assert(countIs(h, 2), "copied");
    bare = new Shared(mk("replaced", 5));
    assert(churn() == 1000);
    assert(countIs(h, 1), "the copy is the only handle left");
    assert(h.value.v == 2 && h.value.tag == "bare", "the copy's block is alive");
    assert(countIs(bare, 1) && bare.value.tag == "replaced");
}

// a `Shared<T> | null` global set to null: the copy is the only handle left
function nullableGlobal() {
    const g = maybe;
    if (g) assert(countIs(g, 2), "a nullable global is born with one count; copied");
    maybe = null;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 3 && g.value.tag == "maybe", "the copy's block is alive");
    }

    assert(g !== null && maybe === null);
}

function main() {
    constGlobal();
    bareGlobal();
    nullableGlobal();
    print("done.");
}
