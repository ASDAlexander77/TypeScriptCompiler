// An element past the end of an array value, without a default, is undefined, as node has it (`const [a, b] = [mk(5)]
// as N[]`: b). Of an array of a reference type (a class, `N | null`, an array) the name is `T | undefined` and reads
// undefined; it read null. Nested patterns, for-of, parameters, in a loop. A
// primitive element (number[], string[]) keeps its type, which cannot hold undefined: it reads as the type's default,
// as before (node: undefined). Every memory model.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function sh(o: N | undefined | null): string {
    return o === undefined ? "u" : o === null ? "null" : "N" + o.x;
}

function param([a, b]: N[]): string {
    return sh(a) + "," + sh(b);
}

function main() {
    let out = "";
    for (let i = 0; i < 50; i++) {
        const [a, b] = [mk(i)] as N[];
        assert(b === undefined, "b undefined");
        out = sh(a) + " " + sh(b);

        const nullable: (N | null)[] = i % 2 == 0 ? [null] : [mk(i)];
        const [c, d] = nullable;
        out += " " + sh(c) + " " + sh(d);

        const nested: N[][] = [[mk(i)]];
        const [[k, l], m] = nested;
        assert(m === undefined, "m undefined");
        out += " " + sh(k) + " " + sh(l);

        const pairs: N[][] = [[mk(1)], [mk(2), mk(3)]];
        for (const [q, r] of pairs) {
            out += " " + sh(q) + sh(r);
        }

        out += " " + param([mk(4)]) + " " + param([]);
    }

    print(out);
    assert(out == "N49 u N49 u N49 u N1u N2N3 N4,u u,u", "objects");

    // primitives keep their type's value
    const ns: number[] = [1];
    const [e, f] = ns;
    assert(e == 1 && f == 0, "number");
    print("done.");
}
