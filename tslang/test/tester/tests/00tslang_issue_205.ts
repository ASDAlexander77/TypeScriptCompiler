// TSLANG 205: sweep bugs - generic functions returning generic function types, generic arrows returned,
// arrows in arrows of an instantiated generic, untyped parameters of interface function fields

// a generic function whose result is a generic function type using its own type parameter
declare function factory<Out>(): <R>(callback: (d: Out) => R) => (x: Out) => R;

function mk<Out>(): <R>(callback: (d: Out) => R) => (x: Out) => R {
    return <R>(cb: (d: Out) => R) => (x: Out) => cb(x);
}

// a generic arrow returned where a generic function type is declared, or nothing is
function mkId(): <R>(cb: () => R) => R {
    return <R>(cb: () => R) => cb();
}

function mkIdUntyped() {
    return <R>(cb: () => R) => cb();
}

function mkNumber(): (cb: () => number) => number {
    const f = <R>(cb: () => R) => cb();
    return f;
}

// an arrow returned from an arrow instantiated for `(cb) => (x: Out) => number`: its result is its own
function mkTyped<Out>(): (callback: (d: Out) => number) => (x: Out) => number {
    return <R>(cb: (d: Out) => R) => (x: Out) => cb(x);
}

interface Handlers {
    consume: (d: number) => void;
    convert: (d: number) => number;
    m(d: number): void;
}

interface Options<Data> {
    consume: (data: Data) => void;
}

let total: number = 0;

function defineOptions<Data>(options: Options<Data>, d: Data): void {
    options.consume(d);
}

// a type parameter given by the literal's other members: a field's result (declared or not), a method's
interface Produced<Data> {
    produce: () => Data;
    consume: (data: Data) => void;
}

function produceConsume<Data>(options: Produced<Data>): Data {
    const value = options.produce();
    options.consume(value);
    return value;
}

interface ProducedMethods<Data> {
    produce(): Data;
    consume(data: Data): void;
}

function produceConsumeMethods<Data>(options: ProducedMethods<Data>): Data {
    const value = options.produce();
    options.consume(value);
    return value;
}

function main() {
    const m = mk<number>();
    const g = m((d: number) => d * 2);
    assert(g(21) == 42, "mk");

    assert(mkId()(() => 7) == 7, "mkId");
    assert(mkIdUntyped()(() => 8) == 8, "mkIdUntyped");
    assert(mkNumber()(() => 9) == 9, "mkNumber");
    assert(mkTyped<number>()((d: number) => d + 1)(41) == 42, "mkTyped");

    // untyped parameters of an interface's function fields take the field's parameter types
    const h: Handlers = {
        consume: (d) => { total += d; },
        convert: (d) => d * 3,
        m(d) { total += d * 10; }
    };
    h.consume(1);
    h.m(2);
    assert(total == 21, "handlers");
    assert(h.convert(5) == 15, "convert");

    defineOptions({ consume: (data) => { total += data; } }, 100);
    assert(total == 121, "defineOptions");

    assert(produceConsume({ produce: () => 42, consume: (data) => { total += data; } }) == 42, "produce");
    assert(produceConsume({ consume: (data) => { total += data; }, produce: (): number => 1.5 }) == 1.5, "produce typed");
    assert(produceConsumeMethods({ produce() { return 10; }, consume(data) { total += data; } }) == 10, "methods");
    assert(total == 174.5, "consumed");

    print("done.");
}
