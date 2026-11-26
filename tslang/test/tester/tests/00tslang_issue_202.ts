// TSLANG 202: a named function expression held by a variable of its name in a namespace (`namespace N { export let f =
// function f() {} }`, or an expando `example.isFoo = function isFoo() {}`): "redefinition of symbol named 'N.f'"
namespace N {
    export let f = function f(v: number) {
        return v + 1;
    };

    export let fact = function fact(n: number): number {
        return n <= 1 ? 1 : n * fact(n - 1);
    };
}

function example() {}

example.isFoo = function isFoo(value: string) {
    return value == "foo";
};

function main() {
    assert(N.f(1) == 2, "a function expression named as its variable");
    assert(N.fact(5) == 120, "...calling itself");
    assert(example.isFoo("foo") && !example.isFoo("bar"), "an expando function expression");
    print("done.");
}
