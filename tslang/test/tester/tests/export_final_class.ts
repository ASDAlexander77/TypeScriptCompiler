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

    // a generator method is exported as any other is: the importer calls it by name
    *upTo(k: int) {
        for (let i = 1; i <= k; i++) {
            yield i;
        }
    }
}
