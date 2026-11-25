// TSLANG 191: a module whose `export =` is a namespace (see import_export_equals.ts)
namespace Lib {
    export function twice(n: number): number {
        return n * 2;
    }

    export const name = "lib";
}

export = Lib;
