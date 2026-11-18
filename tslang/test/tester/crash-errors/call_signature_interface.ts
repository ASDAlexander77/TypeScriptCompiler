// a value of an interface or object type with a call signature and other members is called through the signature
interface Counter {
    (): number;
    step: number;
}

type Named = { (n: number): string; label: string };

declare var counter: Counter;
declare var named: Named;

function next() {
    return counter() + counter.step;
}

function describe() {
    return named(1) + named.label;
}
