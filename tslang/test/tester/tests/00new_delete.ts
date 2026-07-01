type Type1 = TypeOf<1>;

function test_int() {
    let i = new Type1();
    delete i;
}

function test_array() {
    let a: number[] = [];
    a.length = 10;
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
