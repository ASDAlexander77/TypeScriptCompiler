// Bugs found by the tsc tests sweep (TSLANG 195):
// - a union with an object shape tested for truth (`if (u)` of a `number | { x: number }`, `a[k] && a[k].x`): "Cast
//   from number | {x:number} to boolean is not supported"
function truthy(u: number | { x: number }) {
    if (u) {
        return 1;
    }

    return 0;
}

// - a field tested for truth (`if (o.bar)` of a `bar: number | null`) was not narrowed: "can't cast from 'null' to
//   'number'" where `o.bar` was read as a number in the branch
type Foo = { bar: number | null; nested: { b: string | null } };

function fieldTruthy(o: Foo) {
    if (o.bar && o.nested.b) {
        const n: number = o.bar;
        const s: string = o.nested.b;
        return n + s.length;
    }

    return -1;
}

// - an object shape (an object literal, an intersection of shapes) thrown: "Not supported type in throw"; it is
//   thrown as an `any`
type Named = { name: string } & { kind: string };

function throwShape(n: number) {
    const e: Named = { name: "n" + n, kind: "k" };
    throw e;
}

function catchShape() {
    try {
        throwShape(1);
    } catch (x) {
        return (<Named>x).name;
    }

    return "none";
}

function catchLiteral() {
    try {
        throw { name: "lit" };
    } catch (x) {
        return (<{ name: string }>x).name;
    }
}

// - a function returning a union of shapes and then an object literal one of them takes, leaving out a field it
//   may (`{ valid: true }` for `{ valid: boolean; msg?: undefined }`), or one none of them takes: "failed statement"
// - ...and so when the function returning the union is declared after the caller: its return type was taken from
//   `{ valid: true }` alone
type Validated = { valid: boolean; msg?: undefined } | { valid: boolean; msg: string };

function validate(b: boolean) {
    if (b) {
        return utilValidate(false);
    }

    return { valid: true };
}

function validateMore(b: boolean) {
    if (b) {
        return utilValidate(false);
    }

    return { valid: true, extra: 1 };
}

function utilValidate(b: boolean): Validated {
    return b ? { valid: true } : { valid: false, msg: "m" };
}

// recursion still finds its return type
function fact(n: number) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}

function main() {
    assert(truthy(0) == 0 && truthy(2) == 1 && truthy({ x: 1 }) == 1, "a union with a shape tested for truth");
    assert(fieldTruthy({ bar: 3, nested: { b: "yy" } }) == 5 && fieldTruthy({ bar: null, nested: { b: "y" } }) == -1,
        "a field tested for truth");
    assert(catchShape() == "n1", "an intersection of shapes thrown");
    assert(catchLiteral() == "lit", "an object literal thrown");
    assert(!validate(true).valid && validate(true).msg == "m" && validate(false).valid, "a union returned, then a shape");
    assert(!validateMore(true).valid && validateMore(false).valid, "a union returned, then another shape");
    assert(fact(5) == 120, "recursion");

    print("done.");
}
