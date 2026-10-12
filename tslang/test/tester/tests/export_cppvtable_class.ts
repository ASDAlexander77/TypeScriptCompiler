namespace M {

    // A @cpp class exported from one module and used from another: the declaration embedded
    // in the binary (__decls) keeps the decorators, @cpp and each method's @slot, so the importer
    // has the same slots - with no `.instanceOf` - and both modules call the same C++ method for
    // second().

    @cpp
    export class Base {
        value: s32;
        @slot(0) first(): s32;
        @slot(1) second(x: s32): s32;

        // TypeScript's own, not a slot: the importer calls this module's function
        twice(x: s32): s32 {
            return this.second(x) * 2;
        }
    }

    @cpp
    export class Derived extends Base {
        @slot(2) third(x: s32, y: s32): s32;
    }

    // called through this module's numbering
    export function callSecond(object: Base, x: s32): s32 {
        return object.second(x);
    }
}
