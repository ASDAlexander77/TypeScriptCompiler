// -mm=own: a call, in a try whose finally yields, of a method that throws an object out of itself is an error at the
// call, as such a throw written there is
class Failure {
    msg = "x" + 1;
}

class W {
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
        for (const v of g(new W())) print(v);
    } catch (e) {
        print("caught");
    }
}
