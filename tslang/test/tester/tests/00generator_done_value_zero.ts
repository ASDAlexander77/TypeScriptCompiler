// A finished generator's `{ value: undefined, done: true }`, when `value` is a type that cannot
// hold undefined, is zero. It was LLVM undef: the value read back was whatever the register held
// (the last yielded 1 here, garbage in a Set<si32>'s `values().drop(4).next()`).
function* one() {
    yield 1;
}

function main() {
    const it = one();
    const first = it.next();
    assert(first.value == 1 && !first.done);

    const finished = it.next();
    assert(finished.done);
    assert(finished.value == 0);

    const again = it.next();
    assert(again.done);
    assert(again.value == 0);

    print("done.");
}
