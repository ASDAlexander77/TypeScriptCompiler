// TSLANG 191: regressions of TypeScript's compiler tests.
// - an interface extending an interface of one call signature (which as a type is that signature's function type)
// - interfaces of one call signature naming each other (`interface A { (): B }`, `interface B { (): A }`)
// - an interface of one call signature in an intersection (`F & { foo: number }`)
// - a field read through a union of shapes, one of whose fields is `unknown`
// - a tuple given to a union of tuples
interface I6 {
    (): number;
}

interface I7 extends I6 {}

interface E {
    (a: string): string;
}

interface Fx extends E {
    (a: string): string;
}

interface A {
    (): B;
}
interface B {
    (): A;
}

interface F {
    (): string;
}

type S =
    | { type: "string"; value: string }
    | { type: "number"; value: number }
    | { type: "unknown"; value: unknown }
    | { value: undefined };

function defined(s: S): boolean {
    if (s.value !== undefined) {
        return true;
    }

    return false;
}

// (it was "'Never' can't be used as parameter type")
function fooOf(x: F & { foo: number }): number {
    return x.foo;
}

function count(x: [number, string] | [number, string, string]): number {
    return x.length;
}

// a type parameter of the name of a class: `new A()` is the class
class A0 {
    n = 3;
}
function makeA0<A0>(x: A0) {
    return new A0();
}

// a module const of a generic function type given an arrow
const fn1: <T>(x: T) => number = (t) => 1;

// a type parameter whose constraint is literals keeps the literal it is given
interface Targets<V> {
    left: V;
    right: V;
}
type Target = keyof Targets<any>;
type Res<F extends Target, V> = Targets<V>[F];
interface Pick2<F extends Target> {
    s: Res<F, string>;
}
function pickS<F extends Target>(p: Pick2<F>) {
    return p.s;
}

function main() {
    assert(makeA0(1).n == 3, "new of a class a type parameter is named as");
    assert(fn1("x") == 1, "a generic const arrow");
    const p: Pick2<"left"> = { s: "l" };
    assert(pickS(p) == "l", "a literal type argument through a constraint of literals");

    const v1: I7 = () => 7;
    assert(v1() == 7, "extends a call signature");

    const e: Fx = (a: string) => a + "!";
    assert(e("x") == "x!", "extends a call signature with one of its own");

    const a: A | null = null;
    const b: B | null = null;
    assert(a == null && b == null, "call signatures naming each other");


    assert(defined({ type: "string", value: "x" }), "a string value");
    assert(!defined({ value: undefined }), "an undefined value");

    assert(count([1, "a"]) == 2, "two elements");
    assert(count([1, "a", "b"]) == 3, "three elements");

    print("done.");
}
