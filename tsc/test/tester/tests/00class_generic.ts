class B<T>
{
    foo(x: T) {
        assert(x == 1);
    }
}

//class A extends B<number> { }

function main() {
    let x = new B<number>();
    //let x: A = new A();
    x.foo(1); // no error

    print("done.");
}