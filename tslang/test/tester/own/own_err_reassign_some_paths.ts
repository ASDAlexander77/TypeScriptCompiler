// -mm=own rejects: `a` is given a new value after its value moved on one path only, and is read
// after the paths join: its release there would destroy the new value on that path and the moved
// one on the other, and it cannot come before the read. (Without the read, the path that gave it a
// new value releases it before the paths join, spec 30.)
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

    print(a.x);
    print("done.");
}
