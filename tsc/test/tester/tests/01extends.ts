let glb1 = 0;

type ElementOfArray<A> = A extends {
    readonly [n: number]: infer T;
} ? T : never;

type ArrayLike<A> = A extends {
    readonly length: number;
    readonly [n: number]: ElementOfArray<A>;
} ? A : never;

function forEach1<A extends {
    readonly length: number;
    readonly [n: number]: ElementOfArray<A>;
}>(this: A) {
    for (const v of this) {
        print(v);
        glb1++;
    }
}

function forEach2<A>(this: ArrayLike<A>) {
    for (const v of this) {
        print(v);
        glb1++;
    }
}

function main() {
    let a = [1, 2, 3.0];

    forEach1(a);

    forEach2(a);

    assert(glb1 == 6);

    print("done.");
}