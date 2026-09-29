// -mm=own, phase 0: a literal owned once. A string literal or a constant array is an immortal
// global (its release skips it) or a copy (its release destroys it) - either way one owner.
class N { v: number[] = []; }
function main() {
    let count = 0;
    for (let i = 0; i < 100000; i++) {
        let s = "abc";
        const n = new N();
        count += s.length + n.v.length;
    }
    assert(count == 300000);
    print("done.");
}
