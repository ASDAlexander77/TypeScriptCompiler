namespace M {

    // A class implementing an interface with an optional field, one class
    // providing it and one omitting it entirely - the class analogue of
    // export_object_literal_structural_typed_extends_interface_optional.ts,
    // whose object-literal equivalent (getInterfaceCloneFields backfilling
    // instead of omitting a missing optional field) took 4 attempts to fix
    // (see interface-extends-optional-field-clone-bug-fix memory). Exercises
    // whether a real class that simply never declares the optional member
    // gets the same correct "field genuinely absent" clone across a module
    // boundary, rather than backfilled garbage.

    export interface Shape {
        base: number;
        opt?: number;
    }

    export class WithOpt implements Shape {
        base: number = 1;
        opt: number = 5;
    }

    export class WithoutOpt implements Shape {
        base: number = 2;
    }
}
