// An instance of a generic function calling itself needs its return type declared: it is found from the
// body otherwise, and the body needs the result of the call first. Said as that, not as "not all generic
// types can be identified".

function countUp<T>(value: T, n: number) {
    if (n > 0) {
        return countUp(value, n - 1) + 1;
    }

    return 0;
}

function main() {
    print(countUp("x", 3));
    print("done.");
}
