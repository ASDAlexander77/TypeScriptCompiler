// -mm=own, phase 7: async functions. Their locals are ordinary locals of the function the async
// lowering outlines each body and continuation into, and the inference runs on those too. Each
// case reads through a churn() after an await, where a wrong release would have freed.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

async function two() { return 2; }

// a heap local alive across an await
async function local(k: number) {
    const c = new C(k);
    c.v.push(k);
    const v = await two();
    churn();
    return c.x + c.v.length + v;
}

// returns a fresh object
async function make(k: number) {
    const c = new C(k);
    c.v.push(1);
    return c;
}

// a heap parameter read after an await
async function param(c: C) {
    const v = await two();
    churn();
    return c.x + v;
}

async function str(s: string) {
    return s + "!";
}

// a `for await` body is outlined on its own; a string made there is its temporary
async function* numbers() {
    yield 1;
    await two();
    yield 2;
}

async function loop() {
    let t = 0;
    for await (const x of numbers()) {
        const s = "n" + x;
        t += s.length;
    }

    return t;
}

async function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 10;
        t += await local(n);
        const m = await make(n);
        churn();
        t += m.x + m.v.length;
        const p = new C(n);
        t += await param(p);
        const s = await str("ab");
        t += s.length;
    }

    t += await loop();
    assert(t == 450004, "t");
    print("done.");
}
