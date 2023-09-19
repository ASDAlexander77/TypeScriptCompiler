type T00 = unknown & null;  // null
type T01 = unknown & undefined;  // undefined
type T02 = unknown & null & undefined;  // never
type T03 = unknown & string;  // string
type T04 = unknown & string[];  // string[]
type T05 = unknown & unknown;  // unknown
type T06 = unknown & any;  // any

// In a union an unknown absorbs everything

type T10 = unknown | null;  // unknown
type T11 = unknown | undefined;  // unknown
type T12 = unknown | null | undefined;  // unknown
type T13 = unknown | string;  // unknown
type T14 = unknown | string[];  // unknown
type T15 = unknown | unknown;  // unknown
type T16 = unknown | any;  // any

// Type variable and unknown in union and intersection

type T20<T> = T & {};  // T & {}
type T21<T> = T | {};  // T | {}
type T22<T> = T & unknown;  // T
type T23<T> = T | unknown;  // unknown

// unknown in conditional types

type T30<T> = unknown extends T ? true : false;  // Deferred
type T31<T> = T extends unknown ? true : false;  // Deferred (so it distributes)
type T32<T> = never extends T ? true : false;  // true
type T33<T> = T extends never ? true : false;  // Deferred

type T35<T> = T extends unknown ? { x: T } : false;
type T36 = T35<string | number>;  // { x: string } | { x: number }
type T37 = T35<any>;  // { x: any }
type T38 = T35<unknown>;  // { x: unknown }

// keyof unknown

type T40 = keyof any;  // string | number | symbol
type T41 = keyof unknown;  // never

function main() {
    print("done.");
}
