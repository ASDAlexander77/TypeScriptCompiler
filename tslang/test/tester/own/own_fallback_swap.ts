// -mm=own --own-fallback=count: a swap of two elements through a local, `let t = a[i]; a[i] = a[j];
// a[j] = t`. Own refuses it (the local is a second owner of what it read); with the flag the
// elements are counted while they are swapped.
class Node {
    constructor(public v: number, public s: string) {}
}

function swap(a: Node[], i: number, j: number) {
    let t = a[i];
    a[i] = a[j];
    a[j] = t;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const a: Node[] = [];
        for (let k = 0; k < 4; k++) a.push(new Node(i + k, "a" + k));
        swap(a, 0, 3);
        swap(a, 1, 2);
        sum += a[0].v - a[3].v;
        if (a[0].s != "a3") print("bad");
    }

    print(sum);
    print("done.");
}
