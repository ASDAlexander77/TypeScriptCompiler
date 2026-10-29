// A field taken out of the object an owning `let` holds, given `new`'s or a call's own result, that
// nothing sees after the move but the local's end or its next value (spec 65): `return r.value`, `?:`,
// `??`, the read used before it is returned and another field read after, the local given a new object
// after the take, a `let` in a loop, and one given anew in the loop. The field is left empty at the read,
// so the local's release frees nothing of what was taken. Each step churns, so a value freed with its
// object reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

class Holder {
    value = new Item(3);
    other = new Item(9);
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function make() {
    return new Holder();
}

function returned() {
    let r = new Holder();
    return r.value;
}

function chosen(c: boolean, initial: Item) {
    let r = new Holder();
    return c ? initial : r.value;
}

function orDefault(initial?: Item) {
    let r = make();
    return initial ?? r.value;
}

function usedFirst() {
    let r = new Holder();
    const v = r.value;
    churn();
    print(v.n + r.other.n);
    return v;
}

function givenAnew() {
    let r = new Holder();
    const v = r.value;
    r = new Holder();
    churn();
    print(r.value.n);
    return v;
}

function eachNew() {
    const out: Item[] = [];
    for (let i = 0; i < 3; i++) {
        let r = new Holder();
        out.push(r.value);
    }

    return out;
}

function renewedInLoop() {
    const out: Item[] = [];
    let r = new Holder();
    for (let i = 0; i < 3; i++) {
        out.push(r.value);
        r = make();
    }

    return out;
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
    const g = givenAnew();
    churn();
    s += g.n;
    const items = eachNew();
    const more = renewedInLoop();
    churn();
    for (const item of items) s += item.n;
    for (const item of more) s += item.n;
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        if (test() == 3 + 7 + 8 + 3 + 3 + 9 + 9) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
