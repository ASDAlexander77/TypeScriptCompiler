// -mm=own: a call, in a try whose finally yields, of a method whose override in a subclass throws an object out of
// itself is an error at the call, as one of the method itself is (TSLANG 185)
class Failure {
    msg = "x" + 1;
}

class W {
    boom() {
        print("base");
    }
}

class V extends W {
    boom() {
        throw new Failure();
    }
}

function* g(w: W) {
    try {
        yield 1;
        w.boom();
    } finally {
        yield 2;
    }
}

function main() {
    try {
        for (const v of g(new V())) print(v);
    } catch (e) {
        print("caught");
    }
}
