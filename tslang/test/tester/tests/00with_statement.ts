// `with (obj) body`: a name in the body that is a property of obj's static type is that property;
// a variable of the body, a parameter of a function in it, or a name obj has not got, is as it is
class Point {
    x = 1;
    y = 2;
    get sum() { return this.x + this.y; }
    scale(k: number) { this.x *= k; this.y *= k; }
}

interface Named {
    name: string;
}

let y = 100;
let outer = "outer";

function main() {
    const p = new Point();
    with (p) {
        assert(x == 1 && y == 2);
        assert(sum == 3);
        x = 10;
        scale(2);
        assert(outer == "outer");
    }

    assert(p.x == 20 && p.y == 4);
    assert(y == 100);

    let o = { a: 5, b: "b" };
    with (o) {
        a += 1;
        const b = "local";
        assert(b == "local");
        const f = (a: number) => a * 2;
        assert(f(3) == 6);
    }

    assert(o.a == 6 && o.b == "b");

    // a parameter or an inner block's variable hides a property only where it is in scope: after them the
    // names are p's again, not the global `y`
    with (p) {
        const g = (y: number) => y + 1;
        assert(g(1) == 2);
        {
            const x = "inner";
            assert(x == "inner");
        }
        assert(y == 4 && x == 20);
    }

    const n: Named = { name: "n" };
    with (n) {
        with (p) {
            assert(name == "n" && x == 20);
        }
    }

    print("done.");
}
