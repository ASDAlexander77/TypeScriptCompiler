function main() {
    let x: string | number | boolean;

    x = "";
    x; // string

    [x] = [true];
    x; // boolean

    [x = ""] = [1];
    x; // string | number

    ({ x } = { x: true });
    x; // boolean

    ({ y: x } = { y: 1 });
    x; // number

    ({ x = "" } = { x: true });
    x; // string | boolean

    let a: string[] = [];

    for (x of a) {
        x; // string
    }

    print("done.");
}