namespace Generics {

    function swap<T>(arr: T[], i: number, j: number): void {
        let temp: T = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    function sortHelper<T>(arr: T[], callbackfn?: (value1: T, value2: T) => number): T[] {
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

    export function arraySort<T>(arr: T[], callbackfn?: (value1: T, value2: T) => number): T[] {
        return sortHelper(arr, callbackfn);
    }
}

function testGenerics() {
    print("testGenerics")
    // TODO: cast int array to number array
    //let inArray = [4, 3, 4593, 23, 43, -1]
    let inArray = [4.0, 3.0, 4593.0, 23.0, 43.0, -1.0]
    Generics.arraySort(inArray, (x: number, y: number) => { return x - y })
    // TODO: cast int array to number array
    //let expectedArray = [-1, 3, 4, 23, 43, 4593]
    let expectedArray = [-1.0, 3.0, 4.0, 23.0, 43.0, 4593.0]
    for (let i = 0; i < expectedArray.length; i++) {
        assert(inArray[i] == expectedArray[i])
    }
}

function main() {
    testGenerics()
    print("done.");
}
