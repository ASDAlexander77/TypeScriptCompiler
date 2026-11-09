// A computed name that fails to resolve is that error, not a crash (TSLANG_ISSUES 37).
function foo() {
    return 1;
}

class A {
    [foo.x]() {
        return 1;
    }
}

function main() {
    print("not reached");
}
