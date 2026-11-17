// TSLANG_ISSUES 114: an arrow has no `arguments` of its own; it would be the enclosing function's, which the
// arrow would have to capture
function outer(x: number) {
    const f = () => arguments.length;
    return f();
}

function main() {
    print(outer(1));
    print("done.");
}
