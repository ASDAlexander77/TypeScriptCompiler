let c = 0;

function* foo() {
    c++;
    yield 1;
    c++;
    yield 2;
    c++;
}

function main() {
    for (const o of foo()) {
        print(o);
    }

    assert(c == 3);

    print("done.");
}
