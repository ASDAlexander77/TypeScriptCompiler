// An object given as an interface with an index signature is an error at the object ("failed statement").
interface IMap { [s: string]: number; }
function main() {
    let m: IMap = { a: 1 };
}
