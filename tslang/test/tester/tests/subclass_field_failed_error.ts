// A subclass whose field fails is that error: a read of a field it inherits used to crash the
// compiler, as the class had no layout to find the base in.
class Base {
    v = 1;
}

function key() {
    return "k";
}

class Leaf extends Base {
    [key()] = 1;
}

function main() {
    const b = new Leaf();
    const p = b.v;
    print("not reached");
}
