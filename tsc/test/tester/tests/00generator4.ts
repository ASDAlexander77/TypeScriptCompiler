let called = false;

function* foo() {
    print("Hello World");
    called = true;
}

function main() {
    for (const o of foo()) {
        print(o);
    }

    assert(called);

    print("done.");
}
