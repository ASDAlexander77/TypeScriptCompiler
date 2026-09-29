// -mm=own, phase 1 rejects: `a`'s array moves into `b`, and `a` is read after that.
function main() {
    let a: number[] = [1];
    let b = a;
    print(b.length, a.length);
    print("done.");
}
