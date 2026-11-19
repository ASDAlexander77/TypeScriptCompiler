// `a + b` of two `any`s adds as JavaScript does, known at run time: a string on either side joins the
// two as strings, two bigints add as a bigint, anything else adds as numbers
function add(a: any, b: any): any {
    return a + b;
}

function main() {
    assert(<number>add(1, 2) == 3);
    assert(<number>add(1.5, 2) == 3.5);
    assert(<string>add("a", "b") == "ab");
    assert(<string>add("a", 1) == "a1");
    assert(<string>add(2, "b") == "2b");
    assert(<number>add(true, 2) == 3);
    assert(typeof add(1, 2) == "number");
    assert(typeof add("x", 2) == "string");
    assert(typeof add(2n, 3n) == "bigint" && <bigint>add(2n, 3n) == 5n);

    let s: any = "n=";
    const one: any = 1;
    s += one;
    assert(<string>s == "n=1");

    // the other operators were already numbers
    const x: any = 7;
    const y: any = 2;
    assert(x - y == 5);
    assert(x * y == 14);
    assert(y < x);

    print("done.");
}
