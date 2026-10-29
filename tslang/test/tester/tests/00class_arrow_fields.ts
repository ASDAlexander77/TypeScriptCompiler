// A field initialized with an arrow that reads `this` (`fn = () => this.prop`), with no type of its own:
// it is typed by the arrow once the other fields are laid out (`this` the object, as in the
// constructor it runs in), and stored as a function another one can be assigned to.
class A {
    prop = "x";
    n = 1;
    fn = () => this.prop;
    add = (k: number) => this.n + k;
    twice = () => this.fn() + this.fn();

    change() {
        this.fn = () => this.prop + "!";
    }
}

class B extends A {
    more = () => this.add(10);
}

function main() {
    const a = new A();
    assert(a.fn() == "x", "reads this");
    assert(a.add(2) == 3, "with a parameter");
    assert(a.twice() == "xx", "calls another arrow field");

    a.prop = "y";
    assert(a.fn() == "y", "reads this when called");

    a.fn = () => "z";
    assert(a.fn() == "z", "assigned from outside");
    a.change();
    assert(a.fn() == "y!", "assigned in a method");

    const b = new B();
    assert(b.more() == 11, "a subclass's");

    print("done.");
}
