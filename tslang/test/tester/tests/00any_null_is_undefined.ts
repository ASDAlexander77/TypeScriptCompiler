// A null `any` is undefined: a class field of type `any` with no initializer is zeroed memory, a null pointer rather
// than a box, and typeof, print, `+` and the casts read its tag from address 8 and crashed. A number read out of an
// `any` holding undefined is NaN and out of one holding null 0 (it threw "Can't cast from any type"), and an `any`
// holding undefined or null equals no number, string or boolean (`u == 0` threw, `u == false` was true).

class C {
    x: any;
    n = 1;
}

class D extends C {
    y: any;
}

function fields() {
    const c = new C();
    assert(typeof c.x == "undefined", "field typeof");
    assert(c.x === undefined, "field === undefined");
    assert(c.x == null, "field == null");
    assert("[" + c.x + "]" == "[undefined]", "field concat");
    assert(!c.x, "field falsy");
    assert(`${c.x}` == "undefined", "field template");

    const d = new D();
    assert(typeof d.x == "undefined" && typeof d.y == "undefined", "inherited and own fields");

    c.x = "set";
    assert(typeof c.x == "string" && c.x == "set", "field assigned");
}

function numbers() {
    let u: any;
    const fromUndefined = <number>u;
    assert(fromUndefined != fromUndefined, "undefined is NaN");

    const z: any = null;
    const fromNull = <number>z;
    assert(fromNull == 0, "null is 0");

    const c = new C();
    const fromField = <number>c.x;
    assert(fromField != fromField, "a zeroed field is NaN");

    // an integer has no NaN: 0, as `undefined | 0` is
    const intOfUndefined = <s32>u;
    assert(intOfUndefined == 0, "integer of undefined");
    const intOfNull = <s32>z;
    assert(intOfNull == 0, "integer of null");
    const five: any = 5;
    assert(<s32>five == 5, "integer of a number");
}

class K {
    v = 1;
}

function looseUndefined(o: object | null): boolean {
    return o == undefined;
}

function strictUndefined(o: object | null): boolean {
    return o === undefined;
}

function classLooseUndefined(k: K | null): boolean {
    return k == undefined;
}

function stringLooseUndefined(s: string | null): boolean {
    return s != undefined;
}

// loosely, null equals undefined: a value that may be null is `== undefined` when it is (it was false whatever it held)
function nullEqualsUndefined() {
    assert(looseUndefined(null), "null object == undefined");
    assert(!looseUndefined({ a: 1 }), "an object != undefined");
    assert(!strictUndefined(null) && !strictUndefined({ a: 1 }), "strictly, null is not undefined");
    assert(classLooseUndefined(null) && !classLooseUndefined(new K()), "class or null");
    assert(!stringLooseUndefined(null) && stringLooseUndefined("s"), "string or null");
}

// `a == b` and `b == a`, both orders
function equalsNothing(a: any, what: string) {
    assert(!(a == 0) && !(0 == a), what + " == 0");
    assert(a != 0 && 0 != a, what + " != 0");
    assert(!(a === 1) && !(1 === a), what + " === 1");
    assert(a !== 1 && 1 !== a, what + " !== 1");
    assert(!(a == "undefined") && !("undefined" == a), what + " == 'undefined'");
    assert(!(a == "null") && !("null" == a), what + " == 'null'");
    assert(!(a == false) && !(false == a), what + " == false");
    assert(a != false && false != a, what + " != false");
}

function equality() {
    let u: any;
    equalsNothing(u, "undefined");

    const n: any = null;
    equalsNothing(n, "null");

    const c = new C();
    equalsNothing(c.x, "field");

    const five: any = 5;
    assert(five == 5 && 5 == five && five === 5, "a number equals itself");
    assert(!(five != 5) && !(five == 6), "a number is not another");

    const s: any = "x";
    assert(s == "x" && "x" == s && !(s == "y"), "a string");

    const t: any = true;
    assert(t == true && !(t == false), "a boolean");
}

function main() {
    fields();
    numbers();
    equality();
    nullEqualsUndefined();
    print("done.");
}
