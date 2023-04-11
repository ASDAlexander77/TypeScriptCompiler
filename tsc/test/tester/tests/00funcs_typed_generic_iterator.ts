interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

function* g(): Iterable<string> {
    for (let i = 0; i < 10; i++) {
        yield "i = " + i; // string is assignable to string
    }
}

function toArray<X>(xs: Iterable<X>): X[] {
	const arr = [...xs];
    assert(arr.length == 10);
	return arr;
}

function main() {

    let count = 0;

    const iter = g();

    for (const v of iter) {
        count++;
        print(v);
    }

    assert(count == 10);

    const arr = toArray<string>(g());
    assert(arr.length == 10);

    const arr2 = toArray(g());
    assert(arr2.length == 10);

    const arr3 = g().toArray<string>();
    assert(arr3.length == 10);

    const arr4 = g().toArray();
    assert(arr4.length == 10);
    
    print("done.");
}