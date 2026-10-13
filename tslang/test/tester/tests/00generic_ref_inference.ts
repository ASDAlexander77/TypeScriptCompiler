// regression test: a generic function's T is inferred through Ref<T>, also inside a generic tuple
// alias, and through `T | null`.
//
// bug found and fixed: inference had no case for Ref<T>, so `count(v)` with `v: Vec<Item>`
// (`Vec<T> = [first: Ref<T> | null, last: Ref<T> | null]`) failed with "not all types could be
// inferred". And a union was matched member by member in order, so `T | null` against
// `Item | null` could bind T to null.

type Vec<T> = [first: Ref<T> | null, last: Ref<T> | null];
type Item = [id: s32, size: s32];
type Nullable<T> = [value: T | null];

function count<T>(v: Vec<T>): int {
    return <int>((<index><Opaque>v.last - <index><Opaque>v.first) / sizeof<T>());
}

function sizeOfRef<T>(r: Ref<T>): int {
    return sizeof<T>();
}

function sizeOfNullable<T>(n: Nullable<T>): int {
    return sizeof<T>();
}

function main() {
    let items: Item[] = [[1, 2], [3, 4]];
    const first = Ref(items[0]);
    const v: Vec<Item> = [first, <Ref<Item>><Opaque>(<index><Opaque>first + 2 * sizeof<Item>())];

    assert(count<Item>(v) == 2);
    assert(count(v) == 2);

    assert(sizeOfRef(first) == sizeof<Item>());

    const n: Nullable<Item> = [items[1]];
    assert(sizeOfNullable(n) == sizeof<Item>());

    print("done.");
}
