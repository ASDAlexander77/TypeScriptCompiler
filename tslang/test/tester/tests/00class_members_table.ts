// A class's member table (MLIRGenMemberTable.cpp): the class vtable's slot after `.instanceOf` holds the
// address of `<class>.members`. A class read by name (boxed into `any` here) has its public and protected
// members there; one never read by name has an empty table. test-members-table checks the IR.
class P {
    x = 1;
    protected y = 2;
    private z = 3;
    get g() { return 4; }
    m() { return 5; }
    static s = 6;
}

class Q {
    a = 1;
}

function main() {
    const p = new P();
    const a: any = p;
    assert(<number>a["x"] == 1);
    const q = new Q();
    assert(q.a == 1);
    print("done.");
}
