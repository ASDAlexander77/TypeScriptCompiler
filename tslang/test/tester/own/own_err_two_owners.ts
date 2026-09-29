// -mm=own, phase 0 rejects: the array literal would have two owners, the field and `a`. With
// the count check loosened this compiles and double-frees; rc runs it correctly.
class N { v: number[] = [0]; }
function main() {
    let sum = 0;
    for (let i = 0; i < 100000; i++) {
        const n = new N();
        let a = (n.v = [i, i + 1]);
        const filler = [i, i, i, i];
        sum += a[1] - i + filler.length;
    }
    print(sum);
    assert(sum == 500000);
    print("done.");
}
