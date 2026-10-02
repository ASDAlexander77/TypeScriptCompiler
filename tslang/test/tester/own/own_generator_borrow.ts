// -mm=own, phase 7b: a generator over a parameter that holds a block borrows it. The caller owns
// the generator, which may not outlive the argument; the generator's release gives back neither the
// argument nor what it copied out of it (the `for...of` over it). Each case reads through a churn()
// between yields, so an argument freed early reads garbage.
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

// an array parameter, iterated
function* each(a: number[]) {
    for (const v of a) {
        yield v;
    }
}

// a class parameter, its fields yielded
function* fields(c: C) {
    yield c.x;
    yield c.v.length;
}

// a string parameter: what is made from it is the generator's own
function* lengths(s: string) {
    yield s.length;
    yield (s + "!").length;
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n: number = i % 10;
        const a: number[] = [n, n + 1, n + 2];
        for (const v of each(a)) {
            churn();
            t += v;
        }

        const c = new C(n);
        c.v.push(n);
        for (const v of fields(c)) {
            churn();
            t += v;
        }

        const s = "s" + n;
        for (const v of lengths(s)) {
            churn();
            t += v;
        }

        // given up after the first value
        const it = each(a);
        t += it.next().value;

        // two generators over one argument at once
        const g1 = each(a);
        const g2 = each(a);
        t += g1.next().value + g2.next().value;
        churn();
        t += g1.next().value + g2.next().value;

        // the argument is still the caller's
        t += a.length + c.x;
    }

    assert(t == 1180000, "t");
    print("done.");
}
