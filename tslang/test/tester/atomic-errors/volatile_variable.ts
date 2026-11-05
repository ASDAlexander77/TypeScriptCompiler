function f(flag: boolean) {
    let a: i32 = 1;
    return atomicload(ReferenceOf(a), 7, "", 0, flag);
}

function main() {
    print("unreachable");
}
