type int = 1;

function main() {
    const a = 1;
    const aAny = <any>a;
    const b = <int>aAny;
    print(a, b);
    assert(a == b);
    print("done.");
}
