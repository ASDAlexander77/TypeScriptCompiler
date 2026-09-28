// -mm=own, phase 0 rejects: the field would be a second owner of `a`'s array.
class N { v: number[] = []; }
function main() {
    const a: number[] = [1];
    const n = new N();
    n.v = a;
    print(a.length);
    print("done.");
}
