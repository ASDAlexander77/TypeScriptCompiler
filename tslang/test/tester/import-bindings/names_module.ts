// One of each kind of declaration, for import_renamed.ts and import_namespace.ts.
export function add(a: number, b: number) { return a + b; }

export const ANSWER = 42;

export class Point {
    constructor(public x: number, public y: number) {}
    sum() { return this.x + this.y; }
}

export enum Color { Red = 1, Green, Blue }

export interface Named { name: string }

export type Pair = [first: number, second: number];

export class Box<T> {
    constructor(public v: T) {}
}

export function identity<T>(x: T) { return x; }
