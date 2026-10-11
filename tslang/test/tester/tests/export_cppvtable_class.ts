namespace M {

    // A @cppvtable class exported from one module and used from another: the declaration embedded
    // in the binary (__decls) keeps the decorator, so the importer numbers the slots the same way
    // - from 0, with no `.instanceOf` - and both modules call the same C++ method for second().

    @cppvtable
    export class Base {
        value: s32;
        first(): s32;        // slot 0
        second(x: s32): s32; // slot 1
    }

    @cppvtable
    export class Derived extends Base {
        third(x: s32, y: s32): s32; // slot 2
    }

    // called through this module's numbering
    export function callSecond(object: Base, x: s32): s32 {
        return object.second(x);
    }
}
