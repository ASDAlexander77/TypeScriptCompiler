// -mm=own, phase 2 rejects: `b` borrows `a` (which is used again), and is then stored into a
// field - the field would hold a value `a` goes on owning and will free.
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    let a = new C();
    let b = a;
    h.c = b;
    print(a.x);
    print("done.");
}
