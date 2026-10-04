// Constant arrays inside global data keep their elements, and are never freed.
const aa: number[][] = [[1], [2, 3]];
const t: [number, number[]] = [1, [4, 5]];
// inferred, so the inner array stays in global data: a static header over static data
const st = [1, [6, 7]];

function main() {
    assert(aa.length == 2 && aa[1].length == 2 && aa[1][1] == 3, "nested constant array");
    assert(t[1].length == 2 && t[1][0] == 4, "array in a constant tuple");
    for (let i = 0; i < 3; i++) {
        const inner = aa[1];
        assert(inner[0] == 2, "read through a local, repeatedly");
        const sinner = st[1];
        assert(sinner.length == 2 && sinner[1] == 7, "a static header, read through a local repeatedly");
    }
    print("done.");
}
