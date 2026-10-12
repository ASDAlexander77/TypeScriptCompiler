// An error in a derived class's implicit constructor was reported at the start of the file.
// Here the base's constructor is private, so the implicit one can't call it.

class Base {
    private constructor(protected a: int) {
    }
}

class Derived extends Base {
    extra = 1;
}
