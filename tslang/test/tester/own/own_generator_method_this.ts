// -mm=own: a generator method whose state borrows `this` (`@borrows("this")`) leaves the object to
// its caller, who uses it after the loop. Through the vtable of a class that is not `@final` too, and
// through a method that only returns the generator (a forwarder). The state reads the object's
// fields through `this` at each step; a yielded tuple of strings takes copies of them. Each case
// churns the heap between steps, so an object freed early reads garbage.
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

class Bag {
    names: string[] = [];
    counts: number[] = [];

    add(name: string, count: number) {
        this.names.push(name);
        this.counts.push(count);
    }

    @borrows("this")
    private *iter() {
        for (let i = 0; i < this.names.length; i++) {
            const name = this.names[i];
            const count = this.counts[i];
            yield <[name: string, count: number]>[name, count];
        }
    }

    @borrows("this")
    private *iterNames() {
        for (let i = 0; i < this.names.length; i++) {
            const name = this.names[i];
            yield name;
        }
    }

    @borrows("this")
    entries() {
        return this.iter();
    }

    @borrows("this")
    names_() {
        return this.iterNames();
    }

    total() {
        let n = 0;
        for (const [name, count] of this.iter()) {
            churn();
            n += count + name.length;
        }

        return n;
    }
}

@final
class Word {
    constructor(public text: string) {}

    @borrows("this")
    *letters() {
        for (const c of this.text) yield <string>c;
    }
}

function main() {
    const b = new Bag();
    b.add("ab", 1);
    b.add("cde", 2);

    let joined = "";
    for (const [name, count] of b.entries()) {
        churn();
        joined += name + count;
    }

    assert(joined == "ab1cde2");

    let names = "";
    for (const name of b.names_()) {
        churn();
        names += name;
    }

    assert(names == "abcde");
    assert(b.total() == 8);

    // the object is still the caller's after each loop
    b.add("f", 3);
    assert(b.total() == 12);

    const w = new Word("xyz");
    let s = "";
    for (const c of w.letters()) s += c;
    for (const c of w.letters()) s += c;
    assert(s == "xyzxyz" && w.text == "xyz");

    print("done.");
}
