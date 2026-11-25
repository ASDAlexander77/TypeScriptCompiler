// TSLANG 191: a module whose `export =` is a class (see import_export_equals.ts)
class Counter {
    n = 0;

    inc(): number {
        return ++this.n;
    }
}

export = Counter;
