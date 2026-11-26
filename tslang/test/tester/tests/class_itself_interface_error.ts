// TSLANG 204: an error by design (it was an internal error)
interface Constructor<T> {
    new (...args: any[]): T;
}

class A {
    a = 1;
}

function main() {
    const c: Constructor<A> = A;
    print("done.");
}
