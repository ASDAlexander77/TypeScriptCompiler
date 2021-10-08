type int = 1;

function makeRangeIterator(start = 0, end = 10000, step = 1) {
    print("makeRangeIterator.");

    let nextIndex = start;
    let iterationCount = 0;

    const rangeIterator = {
        //next: function () {
	next() {
            print("next...");

            let result: [value: int, done: boolean];
            if (nextIndex < end) {
                result = [nextIndex, false];
                nextIndex += step;
                iterationCount++;
                print("result: ", result.value, result.done);
                return result;
            } else {
                print("result done: ", result.value, result.done);
                result = [iterationCount, true];
            }

            return result;
        },
    };

    print("rangeIterator");
    return rangeIterator;
}

function main() {
    print("begin.");
    let it = makeRangeIterator(1, 10, 2);

    let result = it.next();
    while (!result.done) {
        print(result.value); // 1 3 5 7 9
        result = it.next();
    }

    print("done.");
}
