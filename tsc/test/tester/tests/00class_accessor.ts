class C {
    //_length = 0;
    _length: number;

    constructor() {
        this._length = 10;
    }

    get length() {
        return this._length;
    }
    set length(value: number) {
        this._length = value;
    }
}

function main() {
    const c = new C();
    print(c.length);
    assert(c.length == 10);
    print(c.length);
    c.length = 20;
    print(c.length);
    assert(c.length == 20);
    delete c;

    print("done.");
}
