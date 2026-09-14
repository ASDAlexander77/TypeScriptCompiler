// `typeof f === "function"` names no type of its own, only Opaque. A value whose type is known is
// not narrowed to it: that lost the function type, so the function could not be called in the
// branch, and for a generic function nobody instantiated it referenced a function never emitted.
function one() {
    return 1;
}

function id<T>(x: T) {
    return x;
}

function main() {
    let calls = 0;

    if (typeof one === "function") {
        assert(one() == 1, "plain function is still callable");
        calls++;
    }

    if (typeof id === "function") {
        calls++;
    }

    if (typeof id === "function") {
        assert(id(2) == 2, "generic function is still callable");
        calls++;
    }

    const f = () => 3;
    assert(typeof f === "function" ? f() == 3 : false, "arrow function in ?:");

    // `any` is still narrowed at run time
    let a: any = one;
    if (typeof a === "function") {
        calls++;
    }

    assert(calls == 4, "every branch ran");

    print("done.");
}
