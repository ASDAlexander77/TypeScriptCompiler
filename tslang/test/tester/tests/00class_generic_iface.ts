interface IList<T> {
    data(): T;
    next(): string;
}

class List<U> implements IList<U> {
    data(): U { return 1; }
    next(): string { return "Hello"; };
}

function main() {
    const l = new List<number>();
    assert(l.data() == 1);
    const i = <IList<number>>l;
    assert(i.data() == 1);
    print("done.");
}
