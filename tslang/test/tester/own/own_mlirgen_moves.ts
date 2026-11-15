class C { v = 1; }
class H { c: C | null = null; }
function make(): C {
    const c = new C();
    return c;
}
function main() {
    let c = make();
    const h = new H();
    h.c = c;
    let d = new C();
    d = new C();
    print(h.c!.v, d.v);
    print("done.");
}
