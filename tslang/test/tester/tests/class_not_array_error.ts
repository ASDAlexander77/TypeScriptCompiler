// A class given where an array of constructors is expected: that error, not "failed to legalize operation
// 'ts.ClassRef'" (TSLANG 193).
class A {
    x = 1;
}

function register(classes: { new (): A }[]) {}

function main() {
    register(A);
}
