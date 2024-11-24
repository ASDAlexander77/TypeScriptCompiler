namespace A {

    class Point {
        constructor(public x: number, public y: number) { }
    }

    export var UnitSquare : {
        top: { left: Point, right: Point },
        bottom: { left: Point, right: Point }
    } = { top: { left: null, right: null }, bottom: { left: null, right: null } };
}