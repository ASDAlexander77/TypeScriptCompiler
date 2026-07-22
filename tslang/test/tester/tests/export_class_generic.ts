namespace M {

    // Generic class exported from one module and instantiated with concrete
    // type arguments in another. Each distinct instantiation (Box<number>,
    // Box<string>, Pair<number,string>) needs its own correctly-laid-out
    // fields/methods to be materialized across the module boundary, not just
    // a single generic-erased shape.

    export class Box<T> {
        value: T;

        constructor(v: T) {
            this.value = v;
        }

        get(): T {
            return this.value;
        }

        set(v: T): void {
            this.value = v;
        }
    }

    export class Pair<A, B> {
        first: A;
        second: B;

        constructor(a: A, b: B) {
            this.first = a;
            this.second = b;
        }

        swapDescribe(): string {
            return `${this.second}-${this.first}`;
        }
    }
}
