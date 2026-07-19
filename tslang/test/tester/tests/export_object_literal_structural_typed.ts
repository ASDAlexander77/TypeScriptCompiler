namespace A {

    export interface Counter {
        count: number;
        inc(): void;
    }

    // deliberately typed with a STRUCTURAL type, not the interface: the
    // interface cast then happens in the IMPORTING module against the
    // @dllimport-reconstructed tuple type (see
    // import_object_literal_structural_typed.ts), unlike
    // export_object_literal_with_interface.ts where the cast happens here.
    export var counterObj: { count: number; inc(): void } = { count: 0, inc() { this.count = this.count + 1; } };
}
