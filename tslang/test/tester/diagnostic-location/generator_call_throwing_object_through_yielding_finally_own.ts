// -mm=own: a call, in a try whose finally yields, of a function that throws an object out of itself (here through
// another it calls, declared after the generator) is an error at the call, as such a throw written there is
class Failure {
    msg = "x" + 1;
}

function* g() {
    try {
        yield 1;
        viaMid();
    } finally {
        yield 2;
    }
}

function viaMid() {
    boom();
}

function boom() {
    throw new Failure();
}

function main() {
    try {
        for (const v of g()) print(v);
    } catch (e) {
        print("caught");
    }
}
