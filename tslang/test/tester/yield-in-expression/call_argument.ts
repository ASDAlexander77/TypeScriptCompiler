// The callee is evaluated before its argument, so the argument cannot be a yield.
function* gen() {
    print(yield 1);
}

function main() {
    for (const v of gen()) print(v);
}
