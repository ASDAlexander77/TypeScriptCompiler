// Every way an array lets go of its elements - pop, splice, a shorter length, the array's own
// release - goes through the release of the element's class, whose header test under own is
// `word == 0` first (one owner: free), then the slow path (immortal, extra owners, WeakRefs). No
// fallback here: the release shape is the same in every own program.
class Node {
    constructor(public v: number, public s: string) {}
}

function main() {
    const a: Node[] = [];
    for (let i = 0; i < 1000; i++) a.push(new Node(i, "n" + i));
    let sum: number = 0;
    for (let i = 0; i < 10; i++) sum += a.pop()!.v;
    const cut = a.splice(0, 10);
    sum += cut.length + cut[9].v;
    a.length = 0;
    for (let i = 0; i < 1000; i++) a.push(new Node(i, "m" + i));
    for (const n of a) sum += n.v;
    if (a[999].s != "m999") print("bad");
    print(sum);
    print("done.");
}
