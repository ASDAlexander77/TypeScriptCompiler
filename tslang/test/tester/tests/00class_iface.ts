class S {
    val = 5;

    print() {
        print("Hello", this.val);
        assert(this.val == 5);
    }
}

interface IFace2 {
    print();
}

function main() {
    const s = new S();
    const iface2: IFace2 = s;
    iface2.print();

    const m = iface2.print;
    m();

    const mt: () => void = iface2.print;
    mt();

    print("done.");
}