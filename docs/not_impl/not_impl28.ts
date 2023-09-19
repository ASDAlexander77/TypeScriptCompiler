interface Computed<T> {
    read(): T;
    write(value: T);
}

function foo<T>(x: Computed<T>) { }

enum E1 { X }
enum E2 { X }

// Check that we infer from both a.r and b before fixing T in a.w

function f1<T, U>(a: { w: (x: T) => U; r: () => T; }, b: T): U
{
    return w(b);
}

function main() {

    let s: string;

    // Calls below should infer string for T and then assign that type to the value parameter
    foo({
        read: () => s,
        write: value => s = value
    });
    foo({
        write: value => s = value,
        read: () => s
    });


    let v1: number;
    v1 = f1({ w: x => x, r: () => 0 }, 0);
    v1 = f1({ w: x => x, r: () => 0 }, E1.X);
    v1 = f1({ w: x => x, r: () => E1.X }, 0);

    let v2: E1;
    v2 = f1({ w: x => x, r: () => E1.X }, E1.X);

    //let v3 = f1({ w: x => x, r: () => E1.X }, E2.X);  // Error

}