namespace D {

    // The library side of import_defaultlib_collector.ts. It links the default library as a DLL,
    // so the held strings and the churn are allocated by TypeScriptDefaultLib.dll, while the
    // program that holds them links the default library statically. One collector only if that
    // DLL takes it from the same gc.dll. See tslang/docs/single-gc-collector-design.md.

    export function makeKey(i: number): string {
        return `${i}`.padStart(12, "k");
    }

    export function churn(n: number): number {
        let total = 0;
        for (let j = 0; j < n; j++) {
            // `let`: a `const` in this loop fails to resolve when compiled as an imported module
            let s = "z".repeat(40 + (j % 7));
            total = total + s.length;
        }

        return total;
    }
}
