// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// -mm=own: an object element of an array literal no local holds is a borrow of the literal, as out of a
// named array (user ruling 11-23: no move out of a temporary). A `let` given one in a loop and read after
// it would read an object freed with its literal: refused.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function main() {
    let o: N = [mk(1), mk(2)][1];
    for (let j = 0; j < 2; j++) {
        o = [mk(j), mk(3)][0];
    }

    print(o.x);
}
