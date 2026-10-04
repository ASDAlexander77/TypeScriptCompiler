// Functions pushed, unshifted and spliced into an array of functions (#438): each is cast to the
// element type, and one that cannot be is a compile error, not a crash (builtin-cast/*.ts).
function main() {
    let h: (() => number)[] = [];
    let n: number = 0;
    h.push(() => n);
    h.unshift(() => n + 1);
    h.splice(1, 0, () => n + 2);
    assert(h.length == 3, "three functions");
    assert(h[0]() == 1 && h[1]() == 2 && h[2]() == 0, "each in its place");
    print("done.");
}
