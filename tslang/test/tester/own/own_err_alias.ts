// -mm=own, phase 0 rejects: `b` would be a second owner of `a`'s array.
function main() {
    let a: number[] = [1];
    let b = a;
    print(b.length);
    print("done.");
}
