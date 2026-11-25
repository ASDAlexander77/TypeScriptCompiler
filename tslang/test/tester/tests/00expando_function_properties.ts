// TSLANG 191: properties a function is given after it (`foo.bar = 12`, `foo["baz"] = "ok"`, a const arrow too) - a
// namespace merged with the function, as in TypeScript - and an explicit `namespace f {}` merged with `function f`.
// They were "access expression is not applicable to () => void" and "Can't resolve property 'x' of type".
function foo(n: number): number {
    return n + 1;
}
foo.bar = 12;
const key = "baz";
foo[key] = "ok";
foo.twice = (n: number) => n * 2;

const arrow = () => 3;
arrow.label = "arrow";

function counter(): number {
    counter.calls = counter.calls + 1;
    return counter.calls;
}
counter.calls = 0;

function merged(): number {
    return 1;
}
namespace merged {
    export let x = 5;
    export const s = "a";
}

function main() {
    assert(foo(1) == 2, "the function itself");
    assert(foo.bar == 12, "a property");
    assert(foo.baz == "ok", "a property by a constant key");
    assert(foo.twice(4) == 8, "a function property");
    foo.bar = 13;
    assert(foo.bar == 13, "a property assigned again");
    assert(arrow() == 3 && arrow.label == "arrow", "a const arrow's property");
    counter();
    assert(counter() == 2, "a property the function reads and writes");
    assert(merged() == 1 && merged.x == 5 && merged.s == "a", "a namespace merged with a function");
    merged.x = 6;
    assert(merged.x == 6, "a merged namespace's let assigned");

    print("done.");
}
