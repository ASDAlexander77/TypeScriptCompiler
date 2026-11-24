// Array destructuring past the end of its source, with objects: an element's default is its value where the element
// is missing (`const [a = mk(1), b = mk(2)] = [mk(i)]`: b is mk(2)). The compiler crashed on these (the element read
// past the end of the literal was an undefined value read as an object): a literal, a typed tuple, a nested pattern,
// a for...of pattern, a parameter pattern, an assignment. Under gc and rc (own refuses a borrow of the literal's
// element beside a new default).
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function mk(i: number): N {
    return new N(i);
}

function pick([a = mk(1), b = mk(2)]: N[]): number {
    return a.x + b.x;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 100; i++) {
        const [a = mk(1), b = mk(2)] = [mk(i)];
        total += a.x + b.x;

        const t: [N] = [mk(i)];
        const [e, f = mk(3)] = t;
        total += e.x + f.x;

        const [[g, h = mk(4)]] = [[mk(i)]];
        total += g.x + h.x;

        for (const [p, q = mk(10)] of [[mk(i)]]) {
            total += p.x + q.x;
        }

        let k: N, m: N;
        [k, m = mk(8)] = [mk(i)];
        total += k.x + m.x;

        total += pick([mk(i)]) + pick([]);
    }

    print(total);
    assert(total == 32900, "total");
    print("done.");
}
