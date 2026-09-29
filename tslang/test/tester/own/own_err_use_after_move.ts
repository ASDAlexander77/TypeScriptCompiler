// -mm=own, phase 0 rejects: `b` takes `a`'s instance and gives it up at the end of its block; `a` is used after.
class C { v: number[] = [1,2,3]; }
function main() {
    const a = new C();
    { let b = a; print(b.v.length); }
    const f = [9,9,9,9,9,9,9];
    print(a.v.length);
    print("done.");
}
