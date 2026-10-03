// -mm=own, phase 0 rejects: a field store inside a loop of an object made outside it. (A string
// there is copied instead, spec 22.)
class C { constructor(public x: number) {} }
class N { v: number[] = []; c: C | null = null; }
function main() {
    let k: number = 3;
    const n = new N();
    const c = new C(k);
    for (let i = 0; i < 3; i++) {
        n.c = c;
        const filler = "zzzzzzzz" + i;
        print(n.c!.x, filler);
    }
    print("done.");
}
