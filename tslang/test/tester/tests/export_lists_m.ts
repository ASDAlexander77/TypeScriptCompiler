// TSLANG 191: a module that exports through `export { ... }` lists (see import_export_lists.ts)
function add(a: number, b: number): number {
    return a + b;
}

class Point {
    constructor(public x: number, public y: number) {}

    sum(): number {
        return this.x + this.y;
    }
}

const k = 7;
let counter = 0;

function bump(): number {
    return ++counter;
}

interface Shape {
    area(): number;
}

type Num = number;

enum Color {
    Red,
    Green,
    Blue,
}

function answer(): number {
    return 42;
}

export { add, Point, k as seven, bump, Shape, Num, Color };
export { add as plus };
export { answer as default };
