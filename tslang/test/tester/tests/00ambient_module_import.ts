// TSLANG 191: imports of a `declare module "m" { ... }` the program declares - `import x = require("m")`,
// `import * as x from "m"`, `import { a, b as c } from "m"`, a default import, and a module whose `export =` names a
// class or a namespace. They looked for a file ("can't open file: m.ts").
declare module "shapes" {
    interface Shape {
        area(): number;
    }
    function unit(): number;
    const scale: number;
}

declare module "anything";

declare module "counter" {
    class Counter {
        n: number;
    }
    export = Counter;
}

import * as S from "shapes";
import { unit, Shape, scale as theScale } from "shapes";
import Shapes = require("shapes");
import Counter = require("counter");
import C2 from "counter";
import anything = require("anything");
import * as anyNs from "anything";

class Square implements Shape {
    constructor(public side: number) {}
    area(): number {
        return this.side * this.side;
    }
}

function areaOf(s: S.Shape): number {
    return s.area();
}

function main() {
    assert(areaOf(new Square(3)) == 9, "a type of the module, through its namespace");
    const s: Shapes.Shape = new Square(2);
    assert(s.area() == 4, "a type of the module, through require");
    // (declared only: nothing is called that has no body)
    let f: typeof unit | null = null;
    assert(f == null, "a function of the module, by name");
    let c: Counter | null = null;
    let c2: C2 | null = null;
    assert(c == null && c2 == null, "the class `export =` names");
    let v: number = 0;
    if (v == 1) {
        // (a module with no body: what it gives is an `any`; not run, it has no definition)
        print(anything, anyNs);
    }

    print("done.");
}
