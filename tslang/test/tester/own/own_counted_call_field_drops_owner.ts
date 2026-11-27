// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own: a method that overwrites the field a borrow was read from drops the borrow, through
// the method's summary of the places it overwrites (OWN_DROP_FIELDS_ATTR_NAME) as through a store
// in the caller: the old object is freed, and the borrow would read it.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);

    replace(): void {
        this.c = new C(9);
    }
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 2000; i++) {
        keep.push(new C(77777));
    }

    return keep.length;
}

function main() {
    const h = new H();
    const c = h.c;
    h.replace();
    churn();
    print(c.x);
}
