// -mm=own, phase 1 rejects: `a` moves into `b`, then is read again for `c`.
class C { x: number = 5; }
function main() {
    let a = new C();
    let b = a;
    let c = a;
    print(b.x, c.x);
    print("done.");
}
