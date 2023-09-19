// @strict: true
interface IFace {
    cond0: boolean;
    cond1?: boolean;
}

function main() {

    const a: IFace = { cond0: true };

    print(a.cond0);
    print(a.cond1 == undefined);
    print(a.cond1);

    assert(a.cond0);
    assert(a.cond1 == undefined);

    // a.cond1?.value

    print("done.");
}
