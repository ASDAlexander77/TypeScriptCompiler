function f() {
    let b = true;
    return atomicload(ReferenceOf(b), 7);
}

function main() {
    print("unreachable");
}
