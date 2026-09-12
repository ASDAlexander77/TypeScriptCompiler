// A generator's locals live in a state object that is built and retained before the body has
// ever run, so every reference-typed local starts out as an unwritten field of it. Each case
// here keeps such a field alive across a suspension and reads it on the other side.

function makeNumbers(): number[] {
    return [1, 2, 3];
}

function makeLabel(n: number): string {
    return "n" + n;
}

function* numbersFromLocal() {
    const held = makeNumbers();
    yield held[0];
    yield held[2];
}

function localArraySurvivesSuspension(): number {
    let total = 0;
    for (const v of numbersFromLocal()) {
        total = total + v;
    }

    return total;
}

function* labelsFromLocal() {
    const prefix = makeLabel(1);
    yield prefix;
    yield prefix + "!";
}

function localStringSurvivesSuspension(): string {
    let joined = "";
    for (const s of labelsFromLocal()) {
        joined = joined + s;
    }

    return joined;
}

function* counting() {
    yield 1;
    yield 2;
}

// The inner iterator is itself a reference-typed local of the outer generator.
function* scaling() {
    for (const v of counting()) {
        yield v * 10;
    }
}

function nestedGeneratorsAddUp(): number {
    let total = 0;
    for (const v of scaling()) {
        total = total + v;
    }

    return total;
}

// `yield*` is the same shape written differently.
function* delegating() {
    yield 100;
    yield* counting();
}

function delegationAddsUp(): number {
    let total = 0;
    for (const v of delegating()) {
        total = total + v;
    }

    return total;
}

// A local written on one resumption and read on a later one, so the field is genuinely carried
// by the state object rather than living in a single activation.
function* accumulating() {
    let seen = makeNumbers();
    yield seen.length;
    seen.push(4);
    yield seen.length;
    yield seen[3];
}

function localMutatedBetweenSuspensions(): number {
    let total = 0;
    for (const v of accumulating()) {
        total = total + v;
    }

    return total;
}

function main() {
    assert(localArraySurvivesSuspension() == 4, "local array survives suspension");
    assert(localStringSurvivesSuspension() == "n1n1!", "local string survives suspension");
    assert(nestedGeneratorsAddUp() == 30, "nested generators");
    assert(delegationAddsUp() == 103, "yield* delegation");
    assert(localMutatedBetweenSuspensions() == 11, "local mutated between suspensions");

    print("done.");
}
