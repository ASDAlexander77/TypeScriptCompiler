namespace M {

    // A get/set accessor pair defined in the exporting module, overridden in
    // an importer subclass which calls back into the base accessor via
    // `super.celsius`. `super.<accessor> = value` across a module boundary
    // used to crash MLIR verification (the `this` operand for the setter
    // call was left as a raw by-value ClassStorageType struct instead of
    // being materialized to a pointer via getThisRefOfClass, unlike ordinary
    // super method calls) - fixed by threading isSuperClass through
    // ClassAccessorAccess.

    export class Temperature {
        protected _celsius: number = 0;

        get celsius(): number {
            return this._celsius;
        }

        set celsius(v: number) {
            this._celsius = v;
        }

        get fahrenheit(): number {
            return this._celsius * 9 / 5 + 32;
        }
    }
}
