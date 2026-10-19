// A @final exported class whose base is not exported: its importer (import_final_base_static.ts)
// links against the import library, which has no symbols for Base's members, so a call there of
// a member K inherits goes through the vtable.
class Base {
    bm(): int {
        return 1;
    }

    get s2(): int {
        return 11;
    }
}

@final
export class K extends Base {
    km(): int {
        return 2;
    }
}
