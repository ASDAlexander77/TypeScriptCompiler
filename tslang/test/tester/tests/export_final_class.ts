// The library side of a @final class across a module boundary: the importer calls its methods and
// accessor directly, by name. See import_final_class.ts.
@final
export class Counter {
    n = 0;

    inc(): int {
        return ++this.n;
    }

    get value(): int {
        return this.n;
    }
}
