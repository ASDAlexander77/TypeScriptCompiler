// Shared<T> (spec 23.1): Shared.count after each copy and drop - a handle made, copied, given to a
// function that keeps it, returned from one, reassigned to the block it holds, copied in a loop, held
// by another block that dies, replaced in a block by `s.value = x`, and captured as a parameter by a
// closure and by one that escapes.
class Node {
    v = 0;
}

// a holder rather than a bare array parameter: an array is passed as a value, so a callee's push
// grows the callee's copy and not the caller's
class Keeper {
    kept: Shared<Node>[] = [];
}

function keepOne(s: Shared<Node>, into: Keeper) {
    into.kept.push(s);
}

function dropLast(list: Keeper) {
    list.kept.pop();
}

function make() {
    return new Shared(new Node());
}

// a handle parameter captured by a closure: its cell holds a count, and gives it back
function capture(s: Shared<Node>) {
    let t = 1;
    const g = () => s.value.v + t;
    t = 2;
    return g();
}

// a closure that escapes with a handle parameter: its box holds the count
function reader(s: Shared<Node>) {
    return () => s.value.v;
}

function main() {
    const a = new Shared(new Node());
    assert(Shared.count(a) == 1, "one handle");

    let b = a;
    assert(Shared.count(a) == 2, "copied");

    b = a;
    assert(Shared.count(a) == 2, "reassigned to the block it holds");

    const list = new Keeper();
    keepOne(a, list);
    assert(Shared.count(a) == 3, "kept by a callee");

    b = new Shared(new Node());
    assert(Shared.count(a) == 2 && Shared.count(b) == 1, "one handle moved to another block");

    dropLast(list);
    assert(Shared.count(a) == 1, "the kept one dropped");

    const m = make();
    assert(Shared.count(m) == 1, "returned from a function");

    for (let i = 0; i < 1000; i++) {
        const t = a;
        assert(Shared.count(t) == 2, "copied in a loop");
    }

    assert(Shared.count(a) == 1, "every copy in the loop dropped");

    // the payload is released when the outer block dies
    const inner = new Shared(new Node());
    let outer = new Shared(inner);
    assert(Shared.count(inner) == 2, "an outer block holds inner");
    outer = new Shared(new Shared(new Node()));
    assert(Shared.count(inner) == 1, "the payload released when the outer block died");

    // `s.value = x` releases the old payload and takes a count of the new one
    const other = new Shared(new Node());
    const o2 = new Shared(inner);
    assert(Shared.count(inner) == 2, "o2 holds inner");
    o2.value = other;
    assert(Shared.count(inner) == 1 && Shared.count(other) == 2, "s.value = x hands over");

    capture(a);
    capture(a);
    assert(Shared.count(a) == 1, "a captured parameter's count given back");

    for (let i = 0; i < 100; i++) {
        const r = reader(a);
        assert(r() == 0 && Shared.count(a) == 2, "held by an escaping closure");
    }

    assert(Shared.count(a) == 1, "every escaping closure gave its count back");

    print("done.");
}
