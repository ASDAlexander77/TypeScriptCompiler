namespace M {

    // A class implementing an interface that itself `extends` another
    // interface (multilevel), all defined in the exporting module - the
    // class analogue of the object-literal multilevel-extends-interface
    // coverage (export_object_literal_structural_typed_extends_interface_multilevel.ts),
    // which found real cross-module vtable-offset bugs for the object-literal
    // case. Exercises whether a real `class ... implements Derived` gets the
    // same correct vtable slot assignment across a module boundary.

    export interface Base {
        base: number;
        describe(): string;
    }

    export interface Derived extends Base {
        derived: number;
    }

    export class Impl implements Derived {
        base: number = 1;
        derived: number = 2;

        describe(): string {
            return `base=${this.base},derived=${this.derived}`;
        }
    }
}
