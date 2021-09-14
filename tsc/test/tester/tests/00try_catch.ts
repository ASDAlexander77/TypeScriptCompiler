type int = 1;
function main() {
    print("try/catch");

    let t = 0;

    try {
        throw 1;
    } catch (v: int) {
        print(v);
        v = t;
    }

    assert(v == t);

    print("done.");
}
