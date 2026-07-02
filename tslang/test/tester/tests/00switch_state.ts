function foo(v = 0) {
    switchstate(v);
    print("Hello 0");
    state1: ;
    print("Hello 1");
    state2: ;
    print("Hello 2");
    state3: ;
    print("Hello 3");
}

function main() {
    foo(2);
    print("done.");
}
