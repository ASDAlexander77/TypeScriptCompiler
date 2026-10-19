// A class's OPTIONAL function-typed field (`fn?: () => int`) cannot fill an interface's
// optional function-typed field yet: the slot would point at the shared method cell, a
// thunk calling `this.fn(...)`, so the field would read as present on every instance -
// set or not - and an unset one would be called. Until the cell carries a per-instance
// presence check it is an error, as it was before class fields could fill such a slot.

interface MaybeFn {
    fn?: () => int;
}

class OptField {
    fn?: () => int;
}

function main() {
    const m: MaybeFn = new OptField();
}
