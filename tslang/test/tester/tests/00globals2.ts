function nested_global() {
    // example 1
    var a = 10;

    function f() {
        print("Hello func f", a);
    }

    f();
}

function main() {
    let j = 10;
    var i = j + 10;

    assert(i == 20, "Failed. global i");

    nested_global();

    print("done.");
}
