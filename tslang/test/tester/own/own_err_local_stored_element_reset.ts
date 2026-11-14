// TSLANG_ISSUES 65: a local's value moved into an element, then used, borrows the element, so the
// element given another value before the use frees it.
class V {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

class H {
    items: (V | null)[] = [null, null];
}

export function f(h: H, c: boolean): int {
    let m: V | null = null;
    if (c) {
        m = new V(1);
    }

    h.items[0] = m;
    h.items[0] = null;
    return m ? m.n : 0;
}

function main() { print(f(new H(), true)); }
