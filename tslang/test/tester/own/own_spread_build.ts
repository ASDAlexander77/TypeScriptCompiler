// An array a spread builds: MLIRGen makes it fresh in a local of its own that owns nothing, pushes
// into it, and reads it once, when it is done, into the owning local it is made for. The fresh
// array moves into the builder and on into that local.
function main() {
    const a = [1, 2];
    const b = [...a, ...a];
    assert(b.length == 4 && b[3] == 2);

    const src: number[] = [1.5, 2.5, 3.5];
    const t: number[] = [...src];
    t.push(4.5);
    assert(t.length == 4 && src.length == 3);

    let total = 0;
    for (let i = 0; i < 100; i++) {
        const twice = [...src, ...src];
        total += twice.length;
    }

    assert(total == 600);

    print("done.");
}
