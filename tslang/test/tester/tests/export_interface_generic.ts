namespace M {

    // Cross-module counterpart to 00interface_generic.ts. GenericInterfaceInfo
    // (MLIRGenStore.h) is structurally analogous to GenericClassInfo/
    // GenericFunctionInfo, both of whose cross-module export gaps needed
    // dedicated fixes (PR #280, PR #285) - flagged as a plausible, unverified
    // instance of the same "never routed into declExports" gap in
    // decls-cross-module-declaration-mechanism / generic-function-cross-module-export-fix.

    export interface Box<T> {
        value: T;
        get(): T;
    }

    export interface Pair<A, B> {
        first: A;
        second: B;
        describe(): string;
    }
}
