// The library side of @keeps and @borrows across a module boundary: the importer reads them from
// the declarations (__decls) and calls these knowing them. See import_own_markers.ts.
export class Tag {
    constructor(public name: string) {}
}

@final
export class Holder {
    tag: Tag;

    @keeps("tag")
    constructor(tag: Tag) {
        this.tag = tag;
    }
}

@final
export class Word {
    constructor(public text: string) {}

    @borrows("this")
    *letters() {
        for (const c of this.text) yield <string>c;
    }
}

@borrows("w")
export function *lettersOf(w: Word) {
    for (const c of w.text) yield <string>c;
}
