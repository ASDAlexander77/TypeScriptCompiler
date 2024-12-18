// @strict-null false
class C {
    val: number;

    constructor() {
        this.val = 2;
    }
}

function o(val?: C) {
    print(val?.val);
}

function main() {
    o(new C());
    o(null);
    o();

    print("done.")
}
