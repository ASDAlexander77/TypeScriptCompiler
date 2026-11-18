// A type parameter bounded by its own type (F-bounded): testing the constraint instantiates the type again, which
// went past the instantiation limit. That is not an expansion without end and does not fail the program.
interface Entity<T extends Entity<T>> {
    id: number;
    same(other: T): boolean;
}

interface Person<U extends Person<U>> extends Entity<U> {
    n: number;
}

class Customer implements Person<Customer> {
    constructor(public id: number, public n: number) {}

    same(other: Customer): boolean {
        return this.id == other.id;
    }
}

interface Foo<T extends Foo<T, C>, C> {
    foo(bar: C): C;
}

class Bar implements Foo<Bar, string> {
    foo(bar: string): string {
        return bar + "!";
    }
}

function main() {
    const a = new Customer(1, 10);
    const b = new Customer(1, 20);
    assert(a.same(b) && a.n + b.n == 30, "F-bounded interface");
    assert(new Bar().foo("x") == "x!", "F-bounded with a second parameter");
    print("done.");
}
