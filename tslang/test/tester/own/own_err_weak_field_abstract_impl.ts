// own spec 93: a base's abstract field is an accessor pair, which would read the @weak slot unchecked
class Node { v = 1; }
abstract class A { abstract parent: Node | null; }
class B extends A { @weak parent: Node | null = null; }
function main() { const b = new B(); print(b.parent ? 1 : 0); }
