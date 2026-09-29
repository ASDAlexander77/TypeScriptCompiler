// -mm=own, phase 1 rejects: `a` is assigned after its value moved into `b`. rc's assignment
// releases what the slot holds before storing - the moved value, which `b` owns.
class C { x: number = 5; }
function main() {
    let a = new C();
    let b = a;
    a = new C();
    print(a.x, b.x);
    print("done.");
}
