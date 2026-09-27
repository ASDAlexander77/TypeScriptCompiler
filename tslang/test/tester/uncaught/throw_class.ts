// nothing catches it: the run must end with an error, not a crash of the compiler
class Failure {
    constructor(public message: string) {}
}

function fail() {
    throw new Failure("not caught");
}

function main() {
    print("before");
    fail();
}
