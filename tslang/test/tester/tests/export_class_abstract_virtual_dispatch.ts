namespace M {

    // Single-level abstract class whose CONCRETE method dispatches the
    // still-abstract area() through `this` - the minimal shape of the
    // cross-module vtable-slot-mismatch bug: DeclarationPrinter used to drop
    // the `abstract` modifier when embedding this declaration into the
    // compiled binary (__decls), so the reimporting module treated Shape as
    // concrete and synthesized a `.new` vtable slot the exporting module
    // never had (mlirGenClassNew skips it for abstract classes). That
    // shifted every subsequent slot by one, and describe()'s baked-in
    // slot-1 read for area() landed on `.new` instead - returning whatever
    // garbage happened to sit in XMM0 (a pointer-returning function called
    // through a double-returning signature).

    export abstract class Shape {
        color: string = "red";

        abstract area(): number;

        describe(): string {
            return `${this.color} area=${this.area()}`;
        }
    }
}
