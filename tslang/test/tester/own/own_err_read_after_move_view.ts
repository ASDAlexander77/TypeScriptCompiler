// -mm=own rejects: `c` moved into `h.c`, which is overwritten through an interface view of `h`
// before `c` is read (spec 29).
class C { x: number = 5; }
interface HI { c: C | null; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    const hv: HI = h;
    const c = new C();
    h.c = c;
    hv.c = null;
    print(c.x);
}
