// classes another module takes as values (import_class_values.ts)
export namespace K {
    export class Shape {
        static count = 1;

        constructor(public s: number) {}

        static label() {
            return "shape";
        }

        area() {
            return 0;
        }
    }

    export class Square extends Shape {
        static count = 2;

        static label() {
            return "square";
        }

        area() {
            return this.s * this.s;
        }
    }

    export abstract class Kind {
        static id() {
            return 1;
        }
    }

    export class Concrete extends Kind {
        static id() {
            return 2;
        }
    }

    export function make(c: typeof Shape, s: number) {
        return new c(s);
    }
}
