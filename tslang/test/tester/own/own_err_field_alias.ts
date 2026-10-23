// -mm=own rejects: `a`'s array moved into `n.v`, and `n.v` is overwritten before `a` is read: the
// overwrite frees the array `a` still names (spec 29).
class N { v: number[] = []; }
function main() {
    const a: number[] = [1];
    const n = new N();
    n.v = a;
    n.v = [2];
    print(a.length);
    print("done.");
}
