namespace A {

    export interface Point {
        x: number;
        y: number;
    }

    // valid since Point is exported
    export var Origin: Point = { x: 0, y: 0 };

    interface Point3d extends Point {
        z: number;
    }

    // invalid Point3d is not exported
    export var Origin3d: Point3d = { x: 0, y: 0, z: 0 };

    export interface Counter {
        count: number;
        inc(): void;
    }

    export var counter: Counter = { count: 0, inc() { this.count = this.count + 1; } };
}
