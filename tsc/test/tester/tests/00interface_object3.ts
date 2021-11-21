interface IFace {
    print: () => void;
}

function main() {
    const a: IFace = {
        val: 5,
        print: function () { print("hello", this.val); assert(this.val == 5); }
    };
    a.print();

    const m = a.print;
    m();

    const mt: () => void = a.print;
    mt();

    print("done.");
}