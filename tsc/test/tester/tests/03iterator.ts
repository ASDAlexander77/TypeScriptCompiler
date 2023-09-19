let glb1 = 0;

function f3() {
    const syncGenerator = function* () {
        yield 1;
        yield 2;
    };

    const o = { [Symbol.iterator]: syncGenerator };

    for (const x of o) {
        print(x);
        glb1++;
    }
}

function main() {
    f3();
    assert(glb1 == 2);
    print("done.");
}