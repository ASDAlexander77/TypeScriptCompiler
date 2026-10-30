// -mm=own: two named consts built from the same run-time concatenation are two strings, each with
// its own owner. The optimiser's CSE would merge them into one value with two owners, so under own
// it does not run ahead of inference (the JIT optimises: `-jit` is the run that found it). Each is
// boxed into an `any` argument once, and both are read after a `churn()`.
class Bag {
    keys: any[] = [];
    has(key: any) {
        for (const k of this.keys) if (k == key) return true;
        return false;
    }
}

class C {
    has(a: any) {
        return true;
    }
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i);
    }
}

function main() {
    const c = new C();
    const s1 = "Hel" + "lo";
    print(c.has(s1));
    const s2 = "Hel" + "lo";
    print(c.has(s2));

    const b = new Bag();
    let found = 0;
    for (let i = 0; i < 1000; i++) {
        const k1 = "k" + "ey";
        const k2 = "k" + "ey";
        if (b.has(k1)) found++;
        if (b.has(k2)) found++;
        churn();
        assert(k1 == "key" && k2 == "key");
    }

    assert(found == 0);
    assert(s1 == "Hello" && s2 == "Hello");
    print("done.");
}
