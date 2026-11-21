// A generator is its own iterator, as in JavaScript: `[Symbol.iterator]()` returns it ("field "iterator" can't be
// found"). Not under -mm=own, where a method returning `this` is refused.
function* g() {
    yield 1.5;
    yield 2;
}

function main() {
    const it = g();
    const same = it[Symbol.iterator]();
    assert(same.next().value == 1.5);
    assert(it.next().value == 2);
    print("done.");
}
