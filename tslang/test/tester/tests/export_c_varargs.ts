// The library side of import_c_varargs.ts: a C variadic function exported as a variable, and
// taken as a parameter and a field. Its declaration in __decls names the type
// `CVarArgs<(p0: string) => s32>`, so importers call it the C variadic way too.
@varargs
declare function printf(format: string): s32;

export let cprintf = printf;

export function callWith(f: CVarArgs<(format: string) => s32>) {
    return f("%g\n", 2.5);
}

export class Printer {
    constructor(public f: CVarArgs<(format: string) => s32>) {}
}
