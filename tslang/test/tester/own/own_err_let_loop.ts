// -mm=own, phase 1 rejects: `a` is declared outside the loop and moved inside it, so the
// second iteration would move a value that is already gone (spec 2.6).
class C { x: number = 5; }
function main() {
    let a = new C();
    for (let i = 0; i < 3; i++) {
        let b = a;
        print(b.x);
    }
    print("done.");
}
