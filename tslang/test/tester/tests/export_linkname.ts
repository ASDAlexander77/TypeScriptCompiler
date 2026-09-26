namespace L {

    // @linkname is @dllname under the name that fits a static link; it must rename every kind
    // of symbol @dllname does, and survive the __decls round trip to the importer.

    @linkname("linkname_add")
    export function add(a: number, b: number): number {
        return a + b;
    }

    export function addTwice(a: number, b: number): number {
        return add(a, b) + add(a, b);
    }

    @linkname("linkname_counter")
    export let counter = 40;

    export function bump(): number {
        counter = counter + 2;
        return counter;
    }

    export class Calc {
        constructor(public factor: number) {
        }

        @linkname("linkname_calc_twice")
        static twice(x: number): number {
            return x * 2;
        }

        @linkname("linkname_calc_scale")
        scale(x: number): number {
            return x * this.factor;
        }

        @linkname("linkname_calc_get_double")
        get double(): number {
            return this.factor * 2;
        }

        @linkname("linkname_calc_set_double")
        set double(value: number) {
            this.factor = value / 2;
        }
    }
}
