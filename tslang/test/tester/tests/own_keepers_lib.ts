// The library half of own-keepers-objects.cmake: a keeper - an array only ever pushed onto - and the
// function that keeps a callback in it for good. Another module calls it, so it says so with
// `@immortalizes`, and each caller makes what it passes immortal.
let retained: ((n: number) => void)[] = [];

@immortalizes("handler")
export function register(handler: (n: number) => void): void {
    retained.push(handler);
}

export function runAll(n: number): number {
    for (const cb of retained) {
        cb(n);
    }

    return retained.length;
}
