// A value returned by a generator typed Generator<T, TReturn> with a TReturn other than T is an error at the return:
// next() gives `{ value: T }`, and "end" was cast to a number without a word (it read 0).
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: () => { value: T, done: boolean };
}

function* h(): Generator<number, string> {
    yield 1;
    return "end";
}
