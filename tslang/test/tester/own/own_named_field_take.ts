// A field taken out of an object a `const` names, `new`'s or a call's own, that nothing sees after the
// move but its release (spec 64): `return r.value`, `c ? init : r.value`, `init ?? r.value`, the read used
// before it is returned, another field read after, a push of it, and `new` in a loop; a merge only read
// stays a borrow. The field is left
// empty at the read, so the object's release frees nothing of what was taken. Each step churns, so a
// value freed with its object reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
    other = new Item(9);
}

class Pair {
    constructor(public item: Item, public tag: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function pair(n: number) {
    return new Pair(new Item(n), n);
}

function returned() {
    const r = new Holder();
    return r.value;
}

function chosen(c: boolean, initial: Item) {
    const r = new Holder();
    return c ? initial : r.value;
}

function orDefault(initial?: Item) {
    const r = new Holder();
    return initial ?? r.value;
}

function usedFirst() {
    const r = new Holder();
    const v = r.value;
    churn();
    print(v.n + r.other.n);
    return v;
}

// only read, the merge stays a borrow: the object is used after it
function readOnly(c: boolean, initial: Item) {
    const r = new Holder();
    const x = c ? initial : r.value;
    return x.n + r.other.n;
}

function pushed() {
    const out: Item[] = [];
    for (let i = 0; i < 3; i++) {
        const r = new Holder();
        out.push(r.value);
    }

    return out;
}

function ofCall() {
    const p = pair(8);
    const t = p.tag;
    return p.item.n + t;
}

function test() {
    let s: number = 0;
    const a = returned();
    churn();
    s += a.n;
    const b = chosen(false, new Item(4));
    const c = chosen(true, new Item(4));
    churn();
    s += b.n + c.n;
    const d = orDefault();
    const e = orDefault(new Item(5));
    churn();
    s += d.n + e.n;
    const f = usedFirst();
    churn();
    s += f.n;
    const kept = new Item(1);
    s += readOnly(false, kept) + readOnly(true, kept) + kept.n;
    const items = pushed();
    churn();
    for (const item of items) s += item.n;
    s += ofCall();
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        if (test() == 3 + 7 + 8 + 3 + 12 + 10 + 1 + 9 + 16) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
