namespace C {

    // `export declare` makes the binding visible to importers; it must not make this module
    // re-export the C function it binds.

    @linkname("strlen")
    export declare function len(s: string): index;
}
