// TSLANG_ISSUES 92-124 (second batch): cases from TypeScript's own compiler tests that crashed tslang, failed
// with no error, or that it could not compile.

// 92: a tagged template with no substitutions calls a tag with a rest parameter as any call does; the tag got
// one operand for its two inputs, and the verifier read past the operands
function tag(parts: TemplateStringsArray, ...values: any[]) {
    return parts.length * 10 + values.length;
}

function tagInFunction() {
    return tag `foo`;
}

// 96: `x instanceof Object` is known from the type of `x` ("failed statement" was all it said)
class A96 { a = 0 }
class B96 { b = 0 }
function isObject(obj: A96 & { x: string } | B96) {
    return obj instanceof Object;
}

// 121: a numeric name in an object pattern reads an element of an array
const { 1: second } = [10, 20, 30];

// an untyped `static x` and an untyped parameter property are `any`s (both failed with no error)
class Untyped {
    static x
    static y = 2
    constructor(public p, public q) {}
}

// 114: `arguments` is an any[] of the function's parameters
function argCount(x: number, y: number) {
    return arguments.length;
}

function secondArg(x: number, y: number) {
    return <number>arguments[1];
}

class WithArgs {
    m(a: number, b: number, c: number) {
        return arguments.length;
    }
}

// 115: `super` in an arrow in a method is the base of the method's `this`
class Base115 {
    m() {
        return 1;
    }
}

class Derived115 extends Base115 {
    m() {
        const viaArrow = () => super.m() + 10;
        return viaArrow();
    }
}

// 124: a computed field name that is a constant names its field, and the class's other fields keep their
// initializers (every field of the class kept its zero value)
const key124 = "k";
const method124 = "go";
class Computed {
    ["lit"] = 10;
    [key124] = 20;
    plain = 5;
    [method124]() {
        return 3;
    }
}

function main() {
    assert(tagInFunction() == 10, "a tagged template with no substitutions");
    assert(isObject(new B96()), "instanceof Object of a union of objects");

    assert(second == 20, "a numeric name in an object pattern over an array");

    Untyped.x = 1;
    const u = new Untyped(3, 4);
    assert(<number>Untyped.x == 1 && Untyped.y == 2, "untyped static fields");
    assert(<number>u.p == 3 && <number>u.q == 4, "untyped parameter properties");

    assert(argCount(1, 2) == 2, "arguments.length");
    assert(secondArg(3, 4) == 4, "arguments[1]");
    assert(new WithArgs().m(1, 2, 3) == 3, "a method's arguments leave out its object");

    assert(new Derived115().m() == 11, "super in an arrow in a method");

    const c = new Computed();
    assert(c["lit"] == 10 && c["k"] == 20 && c.k == 20, "computed field names");
    assert(c.plain == 5, "the other fields keep their initializers");
    assert(c.go() == 3, "a computed method name");

    print("done.");
}
