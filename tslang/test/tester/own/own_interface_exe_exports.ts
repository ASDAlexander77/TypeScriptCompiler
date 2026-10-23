// -mm=own: an executable's exports are seen by no other module (USER decision, spec 31), so an
// exported interface, and an exported global that holds a value of it, leave the interface closed
// when the module is built into a program (`--emit=exe`, or the JIT's entry module): `Hold.g.get(a)`
// gets `G.get`'s facts and its result borrows `a`. Built as an object file or a library the same
// module is open and `G.get` is the error it was (own-exe-exports.cmake).
class V {
    constructor(public x: number) {}
}

export interface Getter { get(v: V): V; }
class G implements Getter { get(v: V): V { return v; } }

module Hold {
    export let g: Getter = new G();
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 2000; i++) {
        keep.push(new V(77777));
    }

    return keep.length;
}

function main() {
    const a = new V(5);
    {
        const b = Hold.g.get(a);
        assert(b.x == 5, "the borrowed argument");
    }

    churn();
    assert(a.x == 5, "the argument is still there");
    print(a.x);
    print("done.");
}
