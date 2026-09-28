// Box<number> and Box<string> keep a different type in `v`; reading one as the other crashed.
class Box<T> {
    constructor(public v: T) {}
}

function main() {
    const a = new Box<number>(1);
    const b: Box<string> = a;
    print(b.v);
}
