namespace M {

    // An abstract class implementing an interface concretely (describe,
    // which itself calls the still-abstract area()) - combines two things
    // already proven to work separately cross-module (abstract classes:
    // export_class_abstract.ts; class-implements-interface:
    // export_class_interface.ts) in one type, which is exactly where
    // cross-cutting vtable-layout bugs tend to hide: the concrete subclass's
    // vtable must carry both the abstract-class-inherited slots AND the
    // interface's slots, correctly interleaved, built across a module
    // boundary.

    export interface Describable {
        describe(): string;
    }

    export abstract class Shape implements Describable {
        color: string = "red";

        abstract area(): number;

        describe(): string {
            return `${this.color} area=${this.area()}`;
        }
    }
}
