type int = 1;

function makeRangeIterator(start: int, end: int, step: int) {
    const rangeIterator = {
        nextIndex: start,
        end: end,
        step: step,
        iterationCount: 0,
        next: function () {
            let result: { value: int; done: boolean };
            if (this.nextIndex < this.end) {
                result = { value: this.nextIndex, done: false };
                this.nextIndex += this.step;
                this.iterationCount++;
                return result;
            }
            return { value: this.iterationCount, done: true };
        },
    };

    return rangeIterator;
}

function main() {
    // DO NOT PUT CONST, otherwise you can't edit
    let it = makeRangeIterator(1, 10, 2);

    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
    }

    print("done.");
}
