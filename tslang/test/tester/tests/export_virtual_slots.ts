// A @cpp class whose slots tslang numbers (@virtual), exported: the importer
// (import_virtual_slots.ts) reads its size (@vtablesize) and its destructor from __decls, and numbers
// the @virtual methods of a class extending it after them.
@cpp
export class Shape {
    @virtual area(): s32;
    @destructor destroy(): void;
    @virtual scale(x: s32): s32;
}
