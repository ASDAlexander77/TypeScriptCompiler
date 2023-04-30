namespace Generics {

    function swap<T, I>(arr: T[], i: I, j: I): void {
        let temp: T = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    function sortHelper<T>(arr: T[], callbackfn?: (value1: T, value2: T) => T): T[] {
        if (arr.length <= 0 || !callbackfn) {
            return arr;
        }
        let len = arr.length;
        // simple selection sort.
        for (let i = 0; i < len - 1; ++i) {
            for (let j = i + 1; j < len; ++j) {
                if (callbackfn(arr[i], arr[j]) > 0) {
                    swap(arr, i, j);
                }
            }
        }
        return arr;
    }

    export function arraySort<T>(arr: T[], callbackfn?: (value1: T, value2: T) => T): T[] {
        return sortHelper(arr, callbackfn);
    }
}

function testGenerics() {
    print("testGenerics")
    let inArray = [4, 3, 4593, 23, 43, -1]
    Generics.arraySort(inArray, (x, y) => { return x - y })
    let expectedArray = [-1, 3, 4, 23, 43, 4593]
    for (let i = 0; i < expectedArray.length; i++) {
        assert(inArray[i] == expectedArray[i])
    }
}

function main() {
    testGenerics()
    print("done.");
}
