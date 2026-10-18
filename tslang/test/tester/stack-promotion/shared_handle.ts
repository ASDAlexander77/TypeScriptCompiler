// A Shared<T> handle's count sees when an object is destroyed: under rc and own, an object owning
// memory in a module using Shared stays on the heap, so the count drops when the heap version's
// would.
class Item {
    name = "";
}

class Parent {
    child: Shared<Item> | null = null;
}

function main() {
    const child = new Shared(new Item());
    let p1 = new Parent();
    const p2 = new Parent();
    p1.child = child;
    p2.child = child;
    p1 = new Parent();
    const n = Shared.count(child);
    assert(n < 0 || n == 2, "p1's old Parent let go of the child");
    print("done.");
}
