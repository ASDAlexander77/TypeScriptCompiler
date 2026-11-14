// TSLANG_ISSUES 48: the library own_final_method_main imports - an exported base class whose
// `@final` method reads a number. Its declaration says `@final`, so the importer's calls of it run it.
export class Item { constructor(public n: number) {} }
export class Base {
    items: Item[] = [new Item(1), new Item(2)];
    scale: number = 2.0;
    @final factor(): number { return this.scale; }
}
