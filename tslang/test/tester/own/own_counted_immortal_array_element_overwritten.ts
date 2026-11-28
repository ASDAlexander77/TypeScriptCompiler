// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// TSLANG_ISSUES 61: an array global nothing assigns is immortal, but only its own block: an element
// overwritten frees the old element, so a borrow of it ends there.
class V {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

const G = [new V(1), new V(2)];

export function f(): int {
    const v = G[0];
    G[0] = new V(5);
    return v.n;
}

function main() { print(f()); }
