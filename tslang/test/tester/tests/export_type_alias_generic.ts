namespace M {

    // Cross-module counterpart to 00type_aliases_in_generics.ts. The
    // genericTypeAliasMap entry (MLIRGenStore.h) has no dedicated Info struct
    // (unlike GenericClassInfo/GenericFunctionInfo/GenericInterfaceInfo) - it
    // was the last unverified item flagged in
    // decls-cross-module-declaration-mechanism / generic-interface-cross-module-export-fix
    // as a plausible instance of the same "never routed into declExports" gap.

    export type Box<T> = {
        value: T;
    };

    export type Pair<A, B> = {
        first: A;
        second: B;
    };
}
