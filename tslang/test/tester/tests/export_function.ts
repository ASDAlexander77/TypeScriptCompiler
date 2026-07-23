namespace M {

    // Cross-module counterpart to 01arguments.ts / 00funcs.ts, but for a plain
    // (non-generic) function - every other declaration kind (class, interface,
    // enum, vars) has both a "-generic" and a plain cross-module export/import
    // test pair; function only had "-generic" (import_function_generic.ts).
    // Covers what DeclarationPrinter's printFunction/printParams must round-trip
    // correctly through __decls: a required param, an optional param (`?`), a
    // default-valued param, and a void return.

    export function add(a: number, b: number): number {
        return a + b;
    }

    export function greet(name: string, greeting: string = "Hello"): string {
        return `${greeting}, ${name}!`;
    }

    export function maybeDouble(x: number, factor?: number): number {
        if (factor == undefined) factor = 2;
        return x * factor;
    }

    export function noop(): void {
    }

    // a function in the exporting module calling ANOTHER function declared in
    // the SAME exporting module - exercises that intra-module calls still
    // resolve correctly once this module is compiled as a dynamic-import target
    // rather than compiled standalone (distinct from the importer calling in).
    export function addTwice(a: number, b: number): number {
        return add(a, b) + add(a, b);
    }
}
