namespace A {

    export interface Counter {
        count: number;
        inc(): void;
    }

    // untyped export: infers to a boxed ObjectType at its declaration site
    // (no annotation gives the exporter a plain-tuple shape to fall back to).
    // Regression coverage for the "Bug 1" untyped-object-export-declaration
    // boxing fix: the importer's @dllimport reconstruction must box this
    // back to match, both for casting to an interface AND for reading the
    // global's fields directly (no cast at all).
    export var counterObj = { count: 0, inc() { this.count = this.count + 1; } };
}
