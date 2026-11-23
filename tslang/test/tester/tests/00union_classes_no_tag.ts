// A union of classes (`Sq | Ci`, a subclass, null perhaps) is the object's pointer alone, the class asked of the object
// (`.instanceOf`) where the tag is read (user ruling 2026-11-23): read, narrowed, held, widened, boxed; rc and own
// release the object as the class it is.
class Sq { side = 2; label: string; constructor(i: number) { this.label = "s" + i; } area() { return this.side * this.side; } }
class Ci { r = 1; name: string; tags: string[]; constructor(i: number) { this.name = "c" + i; this.tags = ["t" + i]; } area() { return this.r * 3; } }
class Big extends Ci { extra: string; constructor(i: number) { super(i); this.extra = "x" + i; } area() { return 100; } }
type Shape = Sq | Ci;

function make(i: number): Shape {
    if (i % 3 == 0) return new Sq(i);
    if (i % 3 == 1) return new Ci(i);
    return <Ci>new Big(i);
}

function describe(s: Shape) {
    if (s instanceof Sq) return s.label;
    if (s instanceof Big) return s.extra;
    return s.name;
}

function expected(i: number) { return i % 3 == 0 ? "s" + i : i % 3 == 1 ? "c" + i : "x" + i; }

class Holder { shape: Shape = new Sq(0); }

function main() {
    let total = 0;
    for (let i = 0; i < 9; i++) {
        const s = make(i);
        const d = expected(i);
        assert(describe(s) == d, "made");
        total += s.area();
        const h = new Holder();
        h.shape = s;
        assert(describe(h.shape) == d, "held");
        const n: Shape | null = i % 2 == 0 ? null : make(i);
        if (n !== null) assert(describe(n) == d, "with null");
        if (n) total += 0;
        const w: Shape | string = i % 4 == 0 ? "str" : make(i);
        if (typeof w !== "string") assert(describe(w) == d, "widened");
        const a: any = make(i);
        assert(typeof a == "class" || typeof a == "object", "boxed " + typeof a);
    }
    assert(describe(make(3)) == "s3" && describe(make(4)) == "c4" && describe(make(5)) == "x5", "which");
    print("total", total);
    print("done.");
}
