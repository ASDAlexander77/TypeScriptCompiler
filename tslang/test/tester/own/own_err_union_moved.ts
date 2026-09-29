// -mm=own, phase 3 rejects: `c` is moved into the nullable field through the widening view, then
// read.
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    const c = new C();
    h.c = c;
    print(c.x);
    print("done.");
}
