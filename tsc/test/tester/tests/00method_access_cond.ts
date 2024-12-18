// @strict-null false
class C {
    val: number;

    constructor() {
        this.val = 2;
    }

    print() {
        print("Hello World");
    }

    getVal() {
        return 10;
    }
}

function o(val: C) {
    val?.print();
    print(val?.getVal());
}

function main() {
    o(new C());
    o(null);
    print("done.");
}
