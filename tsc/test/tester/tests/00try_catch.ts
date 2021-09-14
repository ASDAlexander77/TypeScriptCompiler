type int = 1;
function main() {
    print("try/catch");

    let t = 1;

    try {
        throw 1;
    } catch (v: int) {
        print("Hello ", v);
        v = t;
    }

    assert(v == t);

    print("done.");
}
