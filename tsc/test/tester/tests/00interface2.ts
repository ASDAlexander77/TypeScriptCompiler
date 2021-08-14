class S {
    print() {
        print("Hello World");
    }
}

interface IPrn {
    print();
}

function run(iface: IPrn) {
    iface.print();
}

function main() {
    const s = new S();
    let iface = <IPrn>s;
    iface.print();
    run(s);

    print("done.");
}
