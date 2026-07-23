namespace M {

    // Cross-module counterpart to decl_type.ts, but broader - every other
    // declaration kind has both a "-generic" and a plain cross-module test
    // pair; type alias only had "-generic" (import_type_alias_generic.ts).
    // Covers a primitive alias, an object-shape alias, a union alias, an
    // alias-of-alias chain, and a function whose signature uses one of these
    // aliases (so the printed __decls text must resolve the alias by name,
    // not just inline its expansion).

    export type Id = number;

    export type UserId = Id;

    export type Point = {
        x: number;
        y: number;
    };

    export type Status = number | string;

    export function makePoint(x: number, y: number): Point {
        return { x: x, y: y };
    }
}
