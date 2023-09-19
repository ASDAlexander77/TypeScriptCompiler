interface IFace {
    print: () => void;
}

function printExt(this:IFace)
{
	this.print();
}

function main() {
    const a: IFace = {
        val: 5,
        print: function () { print("hello", this.val); assert(this.val == 5); }
    };

    a.printExt();

    const ma = a.printExt;
    ma();

    print("done.");
}
