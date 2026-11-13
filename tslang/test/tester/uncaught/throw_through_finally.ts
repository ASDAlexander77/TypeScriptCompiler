// nothing catches it: the trace under the error starts where it was thrown, through the `finally` blocks
// that throw it again
class Failure {
    constructor(public message: string) {}
}

function fail() {
    throw new Failure("not caught");
}

function inner() {
    try {
        fail();
    } finally {
        print("inner finally");
    }
}

function outer() {
    try {
        inner();
    } finally {
        print("outer finally");
    }
}

function main() {
    print("before");
    outer();
}
