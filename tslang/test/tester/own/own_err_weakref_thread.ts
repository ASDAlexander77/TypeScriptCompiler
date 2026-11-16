// A weak count is not atomic: under -mm=own a program that makes WeakRefs to a class may not share
// what could hold one with another thread (Shared.share, as Thread.spawn and a worker do), where it
// could be freed while this thread reads the count.
class Node {
    v: number = 1;
}

class Box {
    n: Node | null = null;
}

function main() {
    const n = new Node();
    const w = new WeakRef(n);
    const b = new Box();
    Shared.share(b);
    print(w.deref() ? 1 : 0, n.v);
}
