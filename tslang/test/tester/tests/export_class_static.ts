namespace M {

    // Static fields/methods defined in the exporting module, inherited (not
    // shadowed) by a subclass in the importer. TS static members are shared,
    // not per-subclass storage - `Derived.count` and `Base.count` must be the
    // SAME global cell across a module boundary, and `super.increment()`
    // called from a static method in the importer must reach the base's
    // static method and mutate the base's storage in place.

    export class Counter {
        static count: number = 0;
        static label: string = "counter";

        static increment(n: number): void {
            Counter.count = Counter.count + n;
        }

        static describe(): string {
            return `${Counter.label}:${Counter.count}`;
        }
    }
}
