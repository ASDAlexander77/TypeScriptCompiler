// A type predicate whose constraint fails is a compile-time type test that folds to `false`
// (the DefaultLib's `__is` / `__is_any`): the failure is the intended answer, not a warning.
function isOf<V extends T, T>(t: T): t is V { return true; }
function isAny<V extends never>(t: V): t is V { return true; }

function sameType<V, T>(t: T, v: V) {
    if (isOf<V, T>(t)) {
        return "same";
    }

    return "other";
}

function keyKind<K>(key: K) {
    if (isAny(key)) {
        return "any";
    }

    return "typed";
}

function main() {
    print(sameType("a", 1.5));
    print(sameType("a", "b"));
    print(keyKind(1));
}
