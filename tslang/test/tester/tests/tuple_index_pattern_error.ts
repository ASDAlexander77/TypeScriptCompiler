// TSLANG 204: an error by design (it failed with "failed statement" alone)
function main() {
    const t: [number, number] = [1, 2];
    let a: number = 0, b: number = 0, c: number = 0;
    [a, b, c] = t;
    print("done.");
}
