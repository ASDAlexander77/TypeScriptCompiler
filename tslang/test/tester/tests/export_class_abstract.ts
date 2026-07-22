namespace M {

    // 2-level abstract chain (abstract extends abstract) defined entirely in
    // the exporting module - the importer supplies the first concrete
    // override. This is the class analogue of the cross-module optional/vtable
    // interface coverage but for abstract methods specifically: an abstract
    // method's vtable slot has no implementation in the declaring module at
    // all, so the slot must be patched purely from a module that never even
    // sees the base class definition being compiled alongside it.

    export abstract class Shape {
        color: string = "red";

        abstract area(): number;

        describe(): string {
            return `${this.color} area=${this.area()}`;
        }
    }

    export abstract class NamedShape extends Shape {
        name: string = "shape";

        abstract area(): number;

        describe(): string {
            return `${this.name}:${super.describe()}`;
        }
    }
}
