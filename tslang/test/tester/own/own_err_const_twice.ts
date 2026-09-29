// -mm=own, phase 0 rejects: two locals would each own the one instance `a` names.
class C { v: number[] = [1,2,3]; }
function main() {
    const a = new C();
    let b = a;
    let c = a;
    print(b.v.length + c.v.length);
    print("done.");
}
