// -mm=own, phase 2 rejects: `b` borrows `a`, which pins it, so the field store cannot move it.
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    let a = new C();
    let b = a;
    h.c = a;
    print(b.x);
    print("done.");
}
