interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

function* g(): Iterable<string> {
    for (let i = 0; i < 10; i++) {
        yield "i = " + i; // string is assignable to string
    }
}

function toArray<X>(xs: Iterable<X>): X[] {
 	//return [...xs]
	let arr : X[] = [];
	for (const e of xs) arr.push(e);

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

    // TODO: bug
    //const arr2 = toArray(g());
    //assert(arr2.length == 10);

    const arr3 = g().toArray<string>();
    assert(arr3.length == 10);
    
    print("done.");
}