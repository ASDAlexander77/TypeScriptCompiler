type Type1 = 1;

function test_int() {
    let i = new Type1();
    delete i;
}

type int = 1;

function test_array() {
    const a = new int[10]();
    a[0] = 1;
    print(a[0]);

    let b = a;
    b[1] = 2;
    print(b[0], b[1]);

    delete a;
}

function main() {
    test_int();
    test_array();
    print("done.");
}
