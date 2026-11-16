// A program making WeakRefs to a class may still share with other threads what cannot hold one: a
// @final class of numbers. The WeakRef's class is @final too: else a subclass of it could be held in a
// field of any class type.
@final
class Node {
    v: number = 1;
}

@final
class Point {
    x: number = 2;
}

function main() {
    const n = new Node();
    const w = new WeakRef(n);
    const p = new Point();
    Shared.share(p);
    assert(w.deref() ? true : false, "alive");
    assert(p.x == 2, "point");
    print("done.");
}
