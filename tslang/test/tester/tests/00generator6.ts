let glb1 = 0;
let glb2 = 0;

function* g() {
  yield* (function* () {
    yield 1.0;
    yield 2.0;
    yield "3.0";
    yield 4.0;
  })();
}

function main() {
    for (const x of g())
        if (typeof x == "string")
            print("string: ", x, glb1++);
        else if (typeof x == "number")
            print("number: ", x, glb2++);

    assert(glb1 == 1);
    assert(glb2 == 3);

    print("done.");
}