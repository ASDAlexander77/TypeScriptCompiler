type int = TypeOf<1>;

function makeRangeIterator1(start: int, end: int, step: int) {
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

function main1() {
    // DO NOT PUT CONST, otherwise you can't edit
    let it = makeRangeIterator1(1, 10, 2);

    let count = 0;
    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
        count++;
    }

    assert(count == 5);
}

function makeRangeIterator2(start: int, end: int, step: int) {
    const rangeIterator = {
        nextIndex: start,
        end: end,
        step: step,
        iterationCount: 0,
        next: function () {
            let result: [value: int, done: boolean];
            if (this.nextIndex < this.end) {
                result = [this.nextIndex, false];
                this.nextIndex += this.step;
                this.iterationCount++;
            } else {
                result = [this.iterationCount, true];
            }

            return result;
        },
    };

    return rangeIterator;
}

function main2() {
    // DO NOT PUT CONST, otherwise you can't edit
    let it = makeRangeIterator2(1, 10, 2);

    let count = 0;
    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
        count++;
    }

    assert(count == 5);
}

function makeRangeIterator3(start = 0, end = 10000, step = 1) {
    const rangeIterator = {
        nextIndex: start,
        end: end,
        step: step,
        iterationCount: 0,
        next: function () {
            let result: [value: int | undefined, done: boolean];
            if (this.nextIndex < this.end) {
                result = [this.nextIndex, false];
                this.nextIndex += this.step;
                this.iterationCount++;
            } else {
                result = [this.iterationCount, true];
            }

            return result;
        },
    };

    return rangeIterator;
}

function main3() {
    // DO NOT PUT CONST, otherwise you can't edit
    let it = makeRangeIterator3(1, 10, 2);

    let count = 0;
    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
        count++;
    }

    assert(count == 5);
}

function main() {
    main1();
    main2();
    main3();

    print("done.");
}
