// A tagged union carries its payload inline behind a pointer to the payload type's descriptor,
// and retaining or releasing one reads the routine to call out of that descriptor. A union that
// holds nothing yet has a null tag - which is the ordinary case, not a corner: a class instance
// arrives from calloc zeroed, and the first assignment to a union field releases the old value
// before storing the new one.

let glb = 0;

function churn(): number {
    let n = 0;
    for (let i = 0; i < 400; i++) {
        const s = "filler " + i;
        n = n + s.length;
    }

    return n;
}

class Holder {
    tagged: number | string = 0;
}

class Nullable {
    data: number | null = 10;

    // read through a narrowing test, from inside the class - the shape a `number | null` field
    // is written in practice
    value(): number {
        if (this.data !== null) {
            return this.data;
        }

        return -1;
    }
}

function firstAssignmentReleasesNothing() {
    // the release of the old value runs against a zeroed field
    const h = new Holder();
    h.tagged = 4;
    churn();
    assert(<number>h.tagged == 4, "the first assignment reached the field");
}

function reassignedUnionKeepsItsLast() {
    const h = new Holder();
    h.tagged = "one";
    h.tagged = 2;
    h.tagged = "three";
    churn();
    assert(h.tagged == "three", "the last value assigned is the one held");
}

function unionFieldHoldsAString() {
    const h = new Holder();
    h.tagged = "kept" + glb;
    churn();
    assert(h.tagged == "kept0", "a freshly built string in a union field survived");
}

function manyHoldersEachAssignedOnce() {
    // every one of these releases a zeroed union on the way to holding a real value
    let total = 0;
    for (let i = 0; i < 200; i++) {
        const h = new Holder();
        h.tagged = i;
        total = total + <number>h.tagged;
    }

    churn();
    assert(total == 19900, "two hundred unions, each written once");
}

function narrowedUnionFieldReads() {
    const n = new Nullable();
    churn();
    assert(n.value() == 10, "reading the field through a narrowing test");
}

function* mixed() {
    yield* (function* () {
        yield 1.0;
        yield "two";
        yield 3.0;
    })();
}

function generatorYieldingAUnion() {
    let numbers = 0;
    let strings = 0;
    for (const x of mixed()) {
        if (typeof x == "string") {
            strings++;
        } else if (typeof x == "number") {
            numbers++;
        }
    }

    churn();
    assert(numbers == 2, "two numbers came out of the generator");
    assert(strings == 1, "and one string");
}

function main() {
    firstAssignmentReleasesNothing();
    reassignedUnionKeepsItsLast();
    unionFieldHoldsAString();
    manyHoldersEachAssignedOnce();
    narrowedUnionFieldReads();
    generatorYieldingAUnion();

    print("done.");
}
