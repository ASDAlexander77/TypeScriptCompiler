// TSLANG_ISSUES 86-122: cases from TypeScript's own compiler tests that crashed tslang or that it could
// not compile.

// 86: a namespace's code statements run from a global constructor of their own; they were emitted among
// the module's globals, and LLVM crashed on the parentless load
namespace Counter {
    export var count = 10;
    count++;
    export var label = "count " + count;
    export function get() {
        return count;
    }
}

namespace Outer.Inner {
    export var total = 1;
    total += Counter.get();
}

// 87: a function in an object literal takes an untyped parameter as `any`; made generic, it crashed
var holder = {
    f: (a) => 7,
    g(b) {
        return 8;
    },
    h: function (c) {
        return 9;
    }
};

// 89: `new Object()` is `{}`; cast to an interface with no members, it crashed
interface Empty {
}
var emptyObject = <Empty>new Object();

// 90: a binding default over a field the value always has is not used
var { fromLiteral = 500 } = { fromLiteral: 20 };

// 99: an instance field reading a static of its own class made the static's global twice
class Statics {
    static base = 3;
    static "quoted" = 4;
    static 0 = 5;
    fromStatic = Statics.base;
    fromQuoted = Statics["quoted"] + 1;
}

// 109: a function expression with untyped parameters, passed as a callback in a method, was never made
class Board {
    ships: number[] = [1, 2, 3];
    allPositive() {
        return this.ships.every(function (ship) {
            return ship > 0;
        });
    }
}

function everyPositive(values: number[]) {
    return values.every(function (value) {
        return value > 0;
    });
}

// 111: arithmetic and ++/-- on an `any` use the number it holds
var untyped;

// 113: `this` at the module level is undefined
var moduleThis = () => this;

// 116: a parameter's default reads the parameters before it
function scaled(x: number, y = x * 2, z = x + y) {
    return z;
}

class Greeter {
    greet(name: string, text = "hello " + name) {
        return text;
    }
}

// 122: an untyped rest parameter's array pattern reads an any[]
function sumTwo(...[first, second]) {
    return <number>first + <number>second;
}

function main() {
    // 86
    assert(Counter.count == 11, "a namespace's code ran");
    assert(Counter.label == "count 11", "a namespace's variable initialized after its code");
    assert(Outer.Inner.total == 12, "a nested namespace's code ran");

    // 87
    assert(holder.f(1) == 7 && holder.g("x") == 8 && holder.h(true) == 9, "object literal functions with untyped parameters");

    // 88: for...in assigns a variable declared before the loop
    let key: string = "";
    let keys = "";
    const point = { x: 1, y: 2 };
    for (key in point) {
        keys += key;
    }
    assert(keys == "xy", "for...in over an object into a variable declared before it");

    let index;
    let sum = 0;
    const values = [10, 20, 30];
    for (index in values) {
        sum += values[index];
    }
    assert(sum == 60, "for...in over an array into a variable declared before it");

    // 89
    const fresh = new Object();
    assert(emptyObject !== undefined && fresh !== undefined, "new Object()");

    // 90
    assert(fromLiteral == 20, "a binding default not used");
    let { present = 1, absent = 2 } = { present: 5 } as { present?: number; absent?: number };
    assert(present == 5 && absent == 2, "binding defaults over optional fields");

    // 99
    const statics = new Statics();
    assert(statics.fromStatic == 3 && statics.fromQuoted == 5, "instance fields reading statics");
    assert(Statics["0"] == 5, "a static with a numeric name");

    // 109
    assert(new Board().allPositive(), "a function expression callback in a method");
    assert(everyPositive([4, 5]) && !everyPositive([4, -5]), "a function expression callback in a function");

    // 111
    untyped = 1;
    untyped++;
    ++untyped;
    untyped--;
    assert(<number>untyped == 2, "++ and -- on an any");
    let boxed: any = 2.5;
    assert(boxed * 2 == 5 && boxed + 1 == 3.5 && boxed - 1 == 1.5 && boxed > 1, "arithmetic on an any");
    assert("s" + boxed == "s2.5", "an any beside a string");
    let counter: any = 3;
    const before = counter++;
    assert(before == 3 && <number>counter == 4, "postfix ++ on an any gives the old number");

    // 113
    assert(moduleThis() === undefined, "this at the module level");

    // 116
    assert(scaled(2) == 6 && scaled(2, 10) == 12, "a default reading earlier parameters");
    assert(new Greeter().greet("you") == "hello you", "a method's default reading an earlier parameter");

    // 122
    assert(sumTwo(1, 2) == 3, "an untyped rest parameter's array pattern");

    print("done.");
}
