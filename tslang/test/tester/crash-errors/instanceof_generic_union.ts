class Box<T> { constructor(public v: T) {} }
function f(b: Box<number> | Box<string>) {
    if (b instanceof Box) return 1;
    return 0;
}
