namespace M {

    // Cross-module counterpart to 00funcs_generic.ts. GenericFunctionInfo
    // (MLIRGenStore.h) is structurally analogous to GenericClassInfo, whose
    // cross-module export gap took 4 fixes to close (PR #280,
    // class-generic-declaration-export-fix) - per
    // decls-cross-module-declaration-mechanism memory, generic
    // functions/interfaces/type-aliases were flagged as plausible,
    // unverified instances of the exact same gap.

    export function identity<T>(x: T): T {
        return x;
    }

    export function pair<A, B>(a: A, b: B): string {
        return `${a}-${b}`;
    }
}
