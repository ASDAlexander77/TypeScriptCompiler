// A call of an interface's optional function-typed member (`undo?: ...`) lends the object to the call, as a
// call of a member that is not optional does: under -mm=own the object is used again after it (the library's
// Generator `return?` / `throw?`, `g.return(v)` and then `g.next()`), not moved into the method value.

interface Counter {
    n: number;
    add: (x: number) => number;
    undo?: (x: number) => number;
}

function make(): Counter {
    return {
        n: 0,
        add(x: number): number {
            this.n += x;
            return this.n;
        },
        undo(x: number): number {
            this.n -= x;
            return this.n;
        },
    };
}

function main() {
    const c = make();
    c.add(5);
    const r = c.undo(2);
    assert(r == 3, "the optional member called");
    assert(c.add(1) == 4, "the object used again after it");
    assert(c.n == 4, "its field");
    print("done.");
}
