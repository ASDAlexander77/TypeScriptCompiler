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
