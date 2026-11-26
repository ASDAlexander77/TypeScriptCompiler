// TSLANG 204: a destructuring pattern as a catch clause's variable (`catch ({ x })`) failed the clause without an
// error. What was caught (an `any`) is destructured as `let { x } = caught;` at the start of the clause. Not under own:
// a field of an `any` is not read there (cast it to its type first).
function caughtFields() {
    try {
        throw { x: 1, y: "s" };
    } catch ({ x, y }) {
        return x + y;
    }

    return "";
}

function caughtNested() {
    try {
        throw { a: { x: 5 } };
    } catch ({ a: { x } }: any) {
        return x;
    }

    return 0;
}

function caughtRenamed() {
    let result = 0;
    try {
        throw { code: 7 };
    } catch ({ code: c }) {
        result = c;
    } finally {
        result++;
    }

    return result;
}

function main() {
    assert(caughtFields() == "1s", "fields of what was caught");
    assert(caughtNested() == 5, "...nested");
    assert(caughtRenamed() == 8, "...renamed, with a finally");

    // nothing thrown: the clause is not run
    try {
    } catch ({ x }) {
        assert(false, "not caught");
    }

    print("done.");
}
