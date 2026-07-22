namespace M {

    // Cross-module counterpart to 00interface_indexer.ts - an exported
    // interface with an index signature, cast-to from an importer-side class
    // implementation. See cross-module-class-indexer-shared-gap-fix memory:
    // DeclarationPrinter never re-emits a class's index signature for
    // -shared reimport; interfaces may share the same gap since their
    // print() function also never mentions `indexes`/InterfaceIndexInfo.

    export interface Storage {
        [index: number]: string;
    }
}
