class Box<T> { constructor(public v: T) {} }
function f(b: Box<number>) {
    return b instanceof Box;
}
