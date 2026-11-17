// `new` through a value of a union of class types makes whichever class it holds: one of them
// abstract can't be made (abstractClassUnionInstantiation.ts)
class ConcreteA {}
abstract class AbstractA { a: string; }

type ConcretesOrAbstracts = typeof ConcreteA | typeof AbstractA;

function make(c: ConcretesOrAbstracts) {
    return new c();
}

function main() {
    make(ConcreteA);
}
