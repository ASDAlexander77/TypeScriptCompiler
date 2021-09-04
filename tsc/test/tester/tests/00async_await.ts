async function f(a = 1) {
    return a;
}

function main() {
    const v = await f(2);
    print(v);
    assert(v == 2);
    print("done.");
}
