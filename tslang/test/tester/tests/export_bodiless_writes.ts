// TSLANG 190: the other half of import_bodiless_writes.ts - a function that gives an element of an array a new
// value, reached through an object it is given, and then reads its other parameter
export class Box {
    constructor(public s: string) {}
}

export class Holder {
    constructor(public items: Box[]) {}
}

export function clobber(b: Box, h: Holder): string {
    h.items[0] = new Box("new" + h.items.length);
    return b.s;
}
