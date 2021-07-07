class C {
    //_length = 0;
    _length: number;
    _length2: number;
    _length3: number;

    constructor() {
        this._length = 10;
        this._length2 = 11;
        this._length3 = 12;
    }

    get length() {
        return this._length;
    }
    set length(value: number) {
        this._length = value;
    }

    get length2() {
        return this._length2;
    }

    set length3(value: number) {
        this._length3 = value;
        print(value);
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

    assert(c.length2 == 11);
    c.length3 = 30;

    delete c;

    print("done.");
}
