// variables a library assigns after the program starts, read and assigned by the module importing them
export let counter: s32 = 0;
export let title = "first";
export let point = { x: 1, y: 2 };
export let twice = (value: number) => value * 2;

export class Holder {
    constructor(public value: number) {}
}

export let holder = new Holder(1);

export function changeAll() {
    counter = 42;
    title = "second";
    point = { x: 10, y: 20 };
    twice = (value: number) => value * 20;
    holder = new Holder(2);
}

export function counterHere(): s32 {
    return counter;
}

export function titleHere(): string {
    return title;
}

export function pointXHere(): number {
    return point.x;
}
