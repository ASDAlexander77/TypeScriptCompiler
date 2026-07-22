namespace M {

    // A get/set indexer pair defined in the exporting module, overridden in
    // an importer subclass which calls back into the base indexer via
    // `super[index]`. Cross-module counterpart to 00class_indexer_super.ts
    // (same-file) - see super-index-access-gap-fix memory.

    export class Storage {
        protected _val: string = "";

        [index: number]: string;

        get(index: number): string {
            return this._val;
        }

        set(index: number, value: string) {
            this._val = value;
        }
    }
}
