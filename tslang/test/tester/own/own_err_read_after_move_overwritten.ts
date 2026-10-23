// -mm=own rejects: `c` moved into `h.c`, which is overwritten on one path before `c` is read (spec
// 29).
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    const c = new C();
    h.c = c;
    if (c.x > 1) { h.c = null; }
    print(c.x);
}
