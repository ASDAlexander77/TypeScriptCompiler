// TSLANG 204: an error by design (it was an internal error). An interface of a construct signature only is made of the
// class (TSLANG 212, 00tslang_issue_212_ctor_interface.ts); one with other members as well is not
interface Constructor<T> {
    new (...args: any[]): T;
    tag: string;
}

class A {
    a = 1;
}

function main() {
    const c: Constructor<A> = A;
    print("done.");
}
