// `Shared.share(v)` marks every block v reaches as shared between threads (ts.MarkShared), and an
// `@atomic class` marks its new objects and what its fields are given. On one thread nothing
// changes but how the counts are changed, so this checks that marking ends - on a cycle, on a
// literal whose header may be read-only - and that a marked block still counts and frees as
// before: Shared.count reads the count without the bit, and the program runs to the end under
// rc, where a count the mark disturbed would free a block in use. gc and none count nothing and
// report -1.

interface HasV {
    v: number;
}

class Leaf {
    v: number = 1;
}

class Ring {
    next: Ring | undefined;
    name: string;

    constructor(name: string) {
        this.name = name;
    }
}

function makeReader(leaf: Leaf) {
    return () => leaf.v;
}

class Bag {
    // a literal: immortal, left alone
    tag = "bag";
    leaves: Leaf[] = [new Leaf(), new Leaf()];
    names: string[] = ["a", "b"];
    boxed: any = new Leaf();
    viewed: HasV = new Leaf();
    either: Leaf | string = new Leaf();
    maybe: Leaf | undefined = new Leaf();
    reader = makeReader(new Leaf());
    handle = new Shared(new Leaf());
}

@atomic
class Counter {
    hits: number = 0;
    last: Leaf | undefined;
    seen: Leaf[] = [];
}

// a subclass of an @atomic class is @atomic too
class NamedCounter extends Counter {
    label: string = "counter";
}

function countIs(s: Shared<Leaf>, expected: number) {
    const c = Shared.count(s);
    return c < 0 || c == expected;
}

function main() {
    // a cycle: the walk stops at the block it marked first
    const a = new Ring("a");
    const b = new Ring("b");
    a.next = b;
    b.next = a;
    Shared.share(a);
    Shared.share(a);
    assert(a.next.name == "b" && b.next.name == "a", "a marked cycle reads as before");
    b.next = undefined;

    // every kind of field, through descriptors too
    const bag = new Bag();
    assert(countIs(bag.handle, 1), "a handle's count before the mark");
    Shared.share(bag);
    assert(countIs(bag.handle, 1), "the mark leaves the count alone");
    const extra = bag.handle;
    assert(countIs(extra, 2), "a marked handle still counts up");
    assert(bag.leaves[1].v == 1 && bag.names[0] == "a" && bag.viewed.v == 1 && bag.reader() == 1,
        "a marked object reads as before");
    assert(bag.tag == "bag", "a literal is left alone");

    // values that own nothing, and a value marked twice
    Shared.share(1);
    Shared.share("literal");
    Shared.share(bag.leaves);
    Shared.share(bag.leaves);

    // a marked block is still freed: replaced, its last reference goes
    bag.leaves = [new Leaf()];
    bag.maybe = undefined;
    assert(bag.leaves.length == 1, "a field of a marked object takes a new value");

    // an @atomic class: marked when made, and each value its fields are given marked as it lands
    const counter = new NamedCounter();
    counter.hits = counter.hits + 1;
    counter.last = new Leaf();
    // not marked: the barrier sees field stores, not an element added to a field's array. This
    // checks only that the program runs on
    counter.seen.push(new Leaf());
    counter.last = undefined;
    assert(counter.hits == 1 && counter.seen.length == 1 && counter.label == "counter", "an @atomic object reads as before");

    print("done.");
}
