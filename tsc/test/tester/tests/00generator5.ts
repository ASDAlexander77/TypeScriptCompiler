let glb1 = 0;

function* g() {
  yield* (function* () {
    yield 1;
    yield 2;
    yield 3;
  })();
}

function f() {
    for (const x of g()) {
        print(x);
        glb1++;
    }
}

function main() {
    f();

    assert(glb1 == 3);

    print("done.");
}