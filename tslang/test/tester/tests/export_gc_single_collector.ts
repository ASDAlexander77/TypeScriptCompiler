namespace G {

    // The library side of import_gc_single_collector.ts. Both the strings the importer holds
    // and the churn are allocated HERE, so this module's collector is the one that has to run
    // and the one that must be able to see the importer's references. A process with two
    // collectors - one in this library, one in the importer or in TypeScriptRuntime.dll - frees
    // the held strings. See docs/single-gc-collector-design.md.

    export function makeKey(i: number): string {
        return `key-${i}-end`;
    }

    export function churn(n: number): number {
        let total = 0;
        for (let j = 0; j < n; j++) {
            // `let`, not `const`: compiled as an imported module (not as the entry file), a `const`
            // declared in this loop body fails with "can't resolve name: s" - a separate bug.
            let s = `zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-${j}`;
            total = total + s.length;
        }

        return total;
    }
}
