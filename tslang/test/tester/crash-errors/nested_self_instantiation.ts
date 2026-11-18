// each level's name holds the one before it, quoted once more: it doubled per level and never reached the depth limit
interface Foo<T> {
    x: Foo<Foo<T>>;
}

function f(p: Foo<number> | null) {
}
