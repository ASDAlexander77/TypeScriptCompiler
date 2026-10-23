// -mm=own rejects: `c` is moved into the nullable field through the widening view, and the field
// is set to null - freeing `c` - before `c` is read (spec 29).
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    const c = new C();
    h.c = c;
    h.c = null;
    print(c.x);
    print("done.");
}
