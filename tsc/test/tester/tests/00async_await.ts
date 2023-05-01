async function f(a = 1) {
    return a;
}

function main() {
    const v = await f(2);
    print(v);
    assert(v == 2);

    const b = async () => 1;
    const r = await b();
    assert(r == 1);

    print("done.");
}
