// a literal a parameter of a union of string literals does not allow is an error (TS2345)
function show(a: "no" | "yes") {
    print(a);
}

function main() {
    show("maybe");
}
