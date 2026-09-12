// `new C(...)` where `C` is not a class but a value of a *constructor interface* - an interface
// whose only member is a `new` signature. The call goes through that interface's vtable slot,
// and the slot is filled by a method the compiler synthesises for the implementing class: it
// builds the instance, runs the constructor, casts the result to the interface the signature
// returns, and hands that back.
//
// That method is built op by op rather than from a `return` statement, so the retain a return
// performs (section 9.24) never ran for it. The instance arrives as an owned result nobody
// claimed, and section 9.30 releases such a value at the end of the block that produced it -
// which here is the block that returns the interface over it. Every caller of `new C(...)` was
// handed a block that had already been given back.
//
// As with section 9.31's tests, every case builds in one block and reads in another with
// `churn()` in between: that release sits at the END of the producing block, so a read in the
// same block still happens first and sees nothing wrong.
//
// The first four cases discriminate - each one corrupts the heap in three runs out of three
// with the retain removed. The last three are the plain class-to-interface cast with no
// constructor interface in it, which was already correct; they are kept as shape coverage.
//
// See docs/reference-counting-evaluation.md section 9.46.

interface Shape {
    area(): number;
}

interface ShapeConstructor {
    new(size: number): Shape;
}

class Square implements ShapeConstructor {
    size: number;

    constructor(size: number) {
        this.size = size;
    }

    area(): number {
        return this.size * this.size;
    }
}

interface Named {
    label(): string;
}

interface NamedConstructor {
    new(label: string): Named;
}

class Tag implements NamedConstructor {
    text: string;

    constructor(text: string) {
        this.text = text;
    }

    label(): string {
        return this.text;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Square(999);
    }
}

// the shape 01class_new.ts has: a constructor interface bound once, then used to build
function buildSquare(size: number): Shape {
    const Ctor: ShapeConstructor = new Square(0);
    return new Ctor(size);
}

function constructedThroughAConstructorInterface() {
    let s = buildSquare(6);
    churn();

    return s.area();
}

// two instances from the same constructor value, so a shared over-release shows on both
function buildTwo(): number {
    const Ctor: ShapeConstructor = new Square(0);
    let a = new Ctor(3);
    let b = new Ctor(4);
    churn();

    return a.area() + b.area();
}

function twoFromOneConstructor() {
    return buildTwo();
}

// a field the instance owns in turn - freeing the instance frees the string with it
function buildTag(text: string): Named {
    const Ctor: NamedConstructor = new Tag("");
    return new Ctor(text);
}

function constructedInstanceKeepsItsString() {
    let t = buildTag("kept");
    churn();

    return t.label() == "kept" ? 41 : 0;
}

// built in a loop and carried out of it, with every iteration's temporary given back
function buildLast(count: number): Shape {
    const Ctor: ShapeConstructor = new Square(0);
    let last = new Ctor(1);
    for (let i = 2; i <= count; i++) {
        last = new Ctor(i);
    }

    return last;
}

function lastOfALoopSurvives() {
    let s = buildLast(9);
    churn();

    return s.area();
}

// the plain cast on its own, with no constructor interface in sight: a class instance made in
// one block, cast to an interface, and handed back
function asShape(size: number): Shape {
    return new Square(size);
}

function classCastToInterfaceOutlivesItsBlock() {
    let s = asShape(7);
    churn();

    return s.area();
}

// the cast's result kept by somebody else, so the interface is the only thing holding on
let kept: Shape[] = [];

function keepShape(size: number) {
    kept.push(new Square(size));
}

function castInterfaceHeldInAnArray() {
    keepShape(8);
    churn();

    return kept[0].area();
}

// a class local and an interface over it living side by side - one release each, and the
// instance has to survive until both are done
function classAndInterfaceTogether(): number {
    let c = new Square(5);
    let s: Shape = c;
    churn();

    return c.size + s.area();
}

function main() {
    assert(constructedThroughAConstructorInterface() == 36, "an instance built through a constructor interface survives its maker");
    assert(twoFromOneConstructor() == 25, "two instances from one constructor value both survive");
    assert(constructedInstanceKeepsItsString() == 41, "an instance built through a constructor interface keeps what it owns");
    assert(lastOfALoopSurvives() == 81, "the instance a loop carries out is not released with the temporaries");
    assert(classCastToInterfaceOutlivesItsBlock() == 49, "a class cast to an interface outlives the block that made it");
    assert(castInterfaceHeldInAnArray() == 64, "an interface made from a class and kept elsewhere is not released");
    assert(classAndInterfaceTogether() == 30, "a class and an interface over it own the instance together");

    print("done.");
}
