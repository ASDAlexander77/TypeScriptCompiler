function f(op: int) {
    let a: i32 = 1;
    return atomicrmw(op, ReferenceOf(a), 2, 7);
}

function main() {
    print("unreachable");
}
