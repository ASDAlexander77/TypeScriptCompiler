// an async function runs to completion and returns its value, so `Promise<T>` (and
// `PromiseLike<T>`, `Awaited<T>`) is T (docs/not_impl/not_impl7.ts)

async function one(): Promise<number> {
    return 1;
}

async function plusOne(): Promise<number> {
    const i = await one();
    return 1 + i;
}

class C {
    async m(): Promise<string> {
        const i = await plusOne();
        return "m" + i;
    }
}

function main() {
    const a = async (): Promise<number> => 1 + await one();
    const fe = async function (): Promise<number> {
        const i = await one();
        return 10 + i;
    };

    const w: Awaited<Promise<number>> = plusOne();
    const l: PromiseLike<string> = new C().m();

    assert(w == 2, "plusOne");
    assert(a() == 2, "arrow");
    assert(fe() == 11, "function expression");
    assert(l == "m2", "method");

    print("done.");
}
