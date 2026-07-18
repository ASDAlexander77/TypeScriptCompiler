// Infrastructure coverage for docs/object-literal-boxing-design.md PR A: getFields
// look-through, object-spread, and ObjectType->tuple cast on a boxed ObjectType.
// The generator wrapper is the only literal boxed as ObjectType today (BoxAsObject
// flag), so it is used here as the vehicle to exercise these paths ahead of PR B
// (which will box every object literal with a method).

function* gen(start: number) {
    yield start;
    yield start + 1;
}

// gap 2: object spread of a boxed ObjectType value -- must read `step`/`next` (and
// any generator-local fields) via property access rather than hitting the
// llvm_unreachable Default case in the SpreadAssignment TypeSwitch. The spread
// result is a plain tuple snapshot (PR B will change this once literals-with-methods
// are boxed too), so only the original generator's continued advancement is checked.
function main1() {
    const it = gen(10);
    const r0 = it.next();
    assert(r0.value == 10);

    const copy = { ...it };

    const r1 = it.next();
    assert(r1.value == 11);
}

// gap 3: ObjectType -> tuple cast, exercised via an annotated declaration whose
// declared type resolves to a tuple type with a function-typed field matching
// the generator wrapper's `next` field. This is a copy at the annotation boundary
// (documented limitation, see design doc §6) -- only that it doesn't crash and the
// snapshot is correct is checked here.
function main2() {
    const it = gen(20);

    const asTuple: { next: () => { value: number; done: boolean } } = it;
    const r = asTuple.next();
    assert(r.value == 20);
}

// gap 1: MLIRTypeHelper::getFields look-through ObjectType -> storage type fields,
// exercised end-to-end via for...of, which discovers `next` on the iterated
// expression's type.
function main3() {
    let values: number[] = [];
    for (const v of gen(30)) {
        values.push(v);
    }
    assert(values.length == 2);
    assert(values[0] == 30);
    assert(values[1] == 31);
}

function main() {
    main1();
    main2();
    main3();

    print("done.");
}
