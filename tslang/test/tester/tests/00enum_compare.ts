// regression test: two enum values compare as their element type.
//
// bug found and fixed: a comparison whose left operand was still an enum (a variable or parameter
// of the enum type, or `E | undefined`) had no lowering, warned "Not applicable logical operator
// for type: 'E'" and was constant false - `a == a` included. Only a comparison with a member (a
// literal) compared, as an integer.

enum Button { Left = 0, Right = 1, Middle = 2 }

enum Name { A = "a", B = "b" }

class H {
    b: Button | undefined;
}

function eq(a: Button, b: Button) { return a == b; }
function ne(a: Button, b: Button) { return a != b; }
function lt(a: Button, b: Button) { return a < b; }
function optEq(a: Button | undefined, b: Button) { return a == b; }
function optOptEq(a: Button | undefined, b: Button | undefined) { return a === b; }
function nameEq(a: Name, b: Name) { return a == b; }

function main() {
    assert(eq(Button.Right, Button.Right));
    assert(!eq(Button.Right, Button.Left));
    assert(ne(Button.Right, Button.Left));
    assert(!ne(Button.Middle, Button.Middle));
    assert(lt(Button.Left, Button.Middle));
    assert(!lt(Button.Middle, Button.Left));

    assert(optEq(Button.Right, Button.Right));
    assert(!optEq(Button.Left, Button.Right));
    assert(!optEq(undefined, Button.Left));

    assert(optOptEq(Button.Right, Button.Right));
    assert(optOptEq(undefined, undefined));
    assert(!optOptEq(undefined, Button.Left));
    assert(!optOptEq(Button.Left, Button.Middle));

    assert(nameEq(Name.B, Name.B));
    assert(!nameEq(Name.A, Name.B));

    // the shape it was found in: a held button against the button of an event
    const h = new H();
    h.b = Button.Right;
    const held = h.b;
    const other: Button = Button.Right;
    assert(held == Button.Right);
    assert(held == other);
    assert(held === other);
    assert(!(held != other));
    if (held !== undefined) {
        assert(held == other);
    }
    else {
        assert(false);
    }

    h.b = undefined;
    assert(h.b != other);

    print("done.");
}
