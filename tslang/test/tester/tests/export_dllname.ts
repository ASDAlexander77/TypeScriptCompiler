namespace M {

    // @dllname replaces the symbol a function, method, accessor or variable is
    // exported under; the importer only sees the re-printed declaration, so it must
    // resolve the custom name instead of the mangled M.xxx one.

    @dllname("custom_add")
    export function add(a: number, b: number): number {
        return a + b;
    }

    export function addTwice(a: number, b: number): number {
        return add(a, b) + add(a, b);
    }

    @dllname("custom_counter")
    export let counter = 40;

    export function bump(): number {
        counter = counter + 2;
        return counter;
    }

    export class Calc {
        constructor(public factor: number) {
        }

        @dllname("calc_twice")
        static twice(x: number): number {
            return x * 2;
        }

        @dllname("calc_scale")
        scale(x: number): number {
            return x * this.factor;
        }

        @dllname("calc_get_double")
        get double(): number {
            return this.factor * 2;
        }

        @dllname("calc_set_double")
        set double(value: number) {
            this.factor = value / 2;
        }
    }
}
