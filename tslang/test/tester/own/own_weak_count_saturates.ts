// -mm=own: WeakRefs made, copied and dropped, before and after their target's last owner lets go
// (a tombstone, under own). Run, it prints what it prints under gc; read as LLVM IR (own-weak-count-shape.cmake),
// a full weak count on a tombstone stays where it is: the increment does not make the tombstone
// immortal - which deref() would read as alive, with its fields released - and the decrement does
// not take it down, so the tombstone is never freed under a WeakRef left to it.
class Node {
    constructor(public v: number) {}
}

function main() {
    let alive = 0;
    for (let i = 0; i < 100; i++) {
        let n: Node | null = new Node(i);
        const w = new WeakRef(n!);
        const copies: WeakRef<Node>[] = [];
        for (let j = 0; j < 4; j++) copies.push(w);
        if (w.deref()) alive++;
        n = null;
        copies.push(w);
        if (copies.length != 5) print("bad");
    }

    print(alive);
    print("done.");
}
