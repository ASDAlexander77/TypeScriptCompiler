// -mm=own, phase 0 rejects: a closure capturing a heap value (closures are phase 5).
function main() {
    const a: number[] = [1];
    const f = () => a.length;
    print(f());
    print("done.");
}
