// The truthiness of an `any` holding an object - a class instance, an object literal, an array -
// is true, as in JavaScript. It threw "Can't cast from any type".
class P {
    x = 1;
}

function main() {
    const c: any = new P();
    assert(c ? true : false, "class instance");
    assert(!!c, "!! of a class instance");

    const o: any = { x: 1 };
    let seen = false;
    if (o) {
        seen = true;
    }
    assert(seen, "object literal");

    const a: any = [1, 2];
    assert(a ? true : false, "array");

    const z: any = 0;
    assert(!z, "0 is still falsy");
    const e: any = "";
    assert(!e, "empty string is still falsy");
    const n: any = null;
    assert(!n, "null is still falsy");

    print("done.");
}
