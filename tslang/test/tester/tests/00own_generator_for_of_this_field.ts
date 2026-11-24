// TSLANG 182: `for (const s of this.items) yield s` in a generator method (158). Under -mm=own the field is read again
// at each step (USER RULING 2026-11-24), so the loop keeps no borrow of it in the generator's state
class Inner {
    words: string[] = ["x" + 1, "y" + 2, "z" + 3];
}

class B {
    items: string[] = ["a" + 1, "b" + 2, "c" + 3];
    pairs: number[][] = [[1, 2], [3, 4]];
    inner = new Inner();

    *each() {
        for (const s of this.items) {
            if (s == "b2") continue;
            yield s;
        }
    }

    *firstTwo() {
        let k = 0;
        for (const s of this.inner.words) {
            if (k++ == 2) break;
            yield s;
        }
    }

    *sums() {
        for (const [a, b] of this.pairs) yield a + b;
    }

    *assigned() {
        let s: string = "";
        for (s of this.items) yield s.length;
    }
}

function main() {
    let out = "";
    for (const s of new B().each()) out += s;
    for (const s of new B().firstTwo()) out += s;
    assert(out == "a1c3x1y2");

    let t: number = 0;
    for (const n of new B().sums()) t += n;
    for (const n of new B().assigned()) t += n;
    assert(t == 16);

    print("done.");
}
