// TSLANG 189: under -mm=own a yield moves the value out of the record an iterator's next() gave; a value yielded a
// second time without a new next() would be one the caller already has, and stays refused
interface Gen<T> {
    next: (value?: any) => { value: T, done: boolean };
}
class Box { constructor(public s: string) {} }
function* twice(it: Gen<Box>): Gen<Box> { const r = it.next(); yield r.value; yield r.value; }
