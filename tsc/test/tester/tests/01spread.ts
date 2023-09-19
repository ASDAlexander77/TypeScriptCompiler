function call<TS extends unknown[]>(
    handler: (...args: TS) => unknown,
    ...args: TS): void {
    for (const v of args) print(v);
}

function main() {

    // Spread Parameters

    call((...args: number[]) => args[0] + args[1], 4, 2) // ok

    print("done.");
}

