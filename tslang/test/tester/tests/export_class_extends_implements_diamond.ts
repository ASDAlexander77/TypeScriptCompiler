namespace M {

    // A class can only `extends` one class, but can `implements` several
    // interfaces - this is the class-shaped equivalent of the interface
    // diamond coverage (Combined extends Left, Right, #268): a cross-module
    // base class plus two cross-module interfaces, both satisfied by one
    // most-derived class in the importer. Exercises the same per-object
    // vtable-patching machinery from a different angle (class vtable slots
    // for inherited methods interleaved with interface vtable slots for
    // implemented methods).

    export class Base {
        base: number = 1;

        addBase(n: number): void {
            this.base = this.base + n;
        }
    }

    export interface Left {
        left: number;
        addLeft(n: number): void;
    }

    export interface Right {
        right: number;
        addRight(n: number): void;
    }
}
