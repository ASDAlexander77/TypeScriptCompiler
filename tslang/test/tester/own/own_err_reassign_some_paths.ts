// -mm=own rejects: `a` is given a new value after its value moved on one path only, so its release
// at scope exit would destroy the new value on that path and the moved one on the other.
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    let a = new C();
    h.c = a;
    if (h.c.x > 3) {
        a = new C();
        print(a.x);
    }

    print("done.");
}
