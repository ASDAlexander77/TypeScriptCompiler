// -mm=own rejects: `a` moves into the field on one path only, and a field store cannot borrow.
class C { x: number = 5; }
class H { c: C = new C(); }
function main(argc: number) {
    const h = new H();
    const a = new C();
    if (argc > 5) {
        h.c = a;
    }
    print("done.");
}
