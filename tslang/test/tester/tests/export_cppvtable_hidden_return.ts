namespace M {

    // A @cppvtable method returning a record (00cppvtable_hidden_return.ts), exported from one
    // module and called from another: the declaration embedded in the binary (__decls) keeps the
    // declared tuple, and @cppnontrivial, so the importer passes the hidden result pointer too.
    // MSVC only (it ignores @cppnontrivial, which the Itanium ABI needs).

    export type Pair = [first: s32, second: s32];

    @cppvtable
    export class Base {
        value: s32;
        first(): s32;        // slot 0
    }

    @cppvtable
    export class Derived extends Base {
        @cppnontrivial
        pair(x: s32): Pair;  // slot 1: a record, returned through a pointer after `this`
    }

    // called through this module's numbering
    export function callPair(object: Derived, x: s32): Pair {
        return object.pair(x);
    }
}
