// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own: the allocator of a class value frees nothing, so an argument read off an array before it outlives it
// (TSLANG 218) - the constructor may, and a borrow of the array's element used after it is an error, as before.
class Q {
    n: number;
    constructor(arr: any[]) {
        arr.pop();
        this.n = 1;
    }
}

function make<T>(c: new (...args: any[]) => T, a: any[]): T {
    const e = a[0];
    const q = new c(a);
    print(e);
    return q;
}

function main() {
    make(Q, ["x" + 1]);
    print("done.");
}
