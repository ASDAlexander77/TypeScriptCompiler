interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

function* g(): Iterable<string> {
    for (let i = 0; i < 10; i++) {
        yield "i = " + i; // string is assignable to string
    }
}

// TODO: fix me
/*
function toArray<X>(xs: Iterable<X>): X[] {
  return [...xs]
}
*/

function main() {

    let count = 0;

    for (const v of g()) {
        count++;
        print(v);
    }

    assert(count == 10);

    //toArray<string>(g());
    //g().toArray<string>();

    print("done.");
}