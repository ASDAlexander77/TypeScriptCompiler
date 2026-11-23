// TSLANG 174: a closure reading a const declared after it, whose value reads another const declared later ("can't
// resolve name: g"); a namespace `var` declared again ("redefinition of symbol"); an object literal given a class
// whose constructor takes parameter properties (the compiler crashed); a function pushed into an array of functions
// takes its types from the element ("can't assign '() => s32' to '() => number'")

class Point {
    constructor(public x: number, public y: number) {}
}

namespace A {
    export class P {
        constructor(public x: number, public y: number) {}
    }
}

namespace C {
    var p: A.P;
    var p = { x: 4, y: 5 };

    var n = 1;
    var n = 7;

    export function get() {
        return p.x + p.y + n;
    }
}

function main() {
    const f = () => g();
    const k = 5;
    const g = () => k;
    assert(f() == 5);

    assert(C.get() == 16);

    const pt: Point = { x: 2, y: 3 };
    assert(pt.x == 2 && pt.y == 3);

    let i = 0;
    const fs: (() => number)[] = [];
    fs.push(() => i);
    fs.unshift(() => 3);
    i = 2;
    assert(fs[0]() == 3);
    assert(fs[1]() == 2);

    print("done.");
}
