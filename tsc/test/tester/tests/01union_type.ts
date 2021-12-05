function main() {
    let a: number | string;

    a = "Hello";

    if (typeof (a) == "string") {
        print("str val:", a);
        assert(a == "Hello");
    }

    a = 10.0;

    if (typeof (a) == "number") {
        print("num val:", a);
        assert(a == 10.0);
    }

    print("done.")
}