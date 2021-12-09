// TODO: review it if you can fix (seems some param values are not captured)
// ALLOC_ALL_VARS_IN_HEAP and ALLOC_TRAMPOLINE_IN_HEAP must be defined to run it correctly

type int = TypeOf<1>;

function makeRangeIterator(start = 0, end = 10000, step = 1) {
    print("makeRangeIterator.");

    let nextIndex = start;
    let iterationCount = 0;

    const rangeIterator = {
        next() {
            let result: [value: int, done: boolean];
            if (nextIndex < end) {
                result = [nextIndex, false];
                nextIndex += step;
                iterationCount++;
                return result;
            } else {
                result = [iterationCount, true];
            }

            return result;
        },
    };

    return rangeIterator;
}

function main() {
    let it = makeRangeIterator(1, 10, 2);

    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
    }

    print("done.");
}
