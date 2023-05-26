class SomeBase {
    public static publicStaticMember = 0;
}

class SomeDerived3 extends SomeBase {
    static fn() {
        super.publicStaticMember = 3;
    }
    static get a() {
        return '';
    }
    static set a(n) {
    }
}

function main() {

    SomeDerived3.fn();
    assert(SomeBase.publicStaticMember == 3);

    print("done.");
}
