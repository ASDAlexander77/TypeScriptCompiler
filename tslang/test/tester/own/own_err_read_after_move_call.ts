// -mm=own rejects: `c` moved into `h.c`, and a call given `h` - whose facts say it may drop - comes
// before `c` is read (spec 29).
class C { x: number = 5; }
class H { c: C | null = null; }
function clear(h: H) { h.c = null; }
function main() {
    const h = new H();
    const c = new C();
    h.c = c;
    clear(h);
    print(c.x);
}
