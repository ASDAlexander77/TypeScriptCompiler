// -mm=own: what an iterator's next() returns is the caller's, whoever implements it - through an
// interface too, where this module cannot see the function. A generator's next moves what it
// yields into the record, and any other next is a function whose callers are not all known, which
// may not return a borrow.
class C {
    constructor(public x: number) {}
}

interface Source {
    next(): { value: C; done: boolean };
}

class Counter implements Source {
    n = 0;

    next() {
        this.n++;
        return { value: new C(this.n), done: this.n > 3 };
    }
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new C(999));
    }
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        const it: Source = new Counter();
        let r = it.next();
        while (!r.done) {
            churn();
            t += r.value.x;
            r = it.next();
        }
    }

    assert(t == 6000, "t");
    print("done.");
}
