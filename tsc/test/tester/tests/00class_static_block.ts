class C {
    static x: number;
    static {
        C.x = 1;
    }
}

function main() {
    print(C.x);
    assert(C.x == 1);
    print("done.");
}
