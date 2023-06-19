function f0() {
    let [] = [1, "hello"];
    let [x] = [1, "hello"];
    let [x, y] = [1, "hello"];
    //let [x, y, z] = [1, "hello"];
    let [, , x] = [0, 1, 2];
    //let x: number;
    //let y: string;
}

function f1() {
    let a = [1, "hello"];
    let [x] = a;
    let [x, y] = a;
    //let [x, y, z] = a;
    //let x: number | string;
    //let y: number | string;
    //let z: number | string;
}

function f2() {
    //let { } = { x: 5, y: "hello" };       // Error, no x and y in target
    //let { x } = { x: 5, y: "hello" };     // Error, no y in target
    //let { y } = { x: 5, y: "hello" };     // Error, no x in target
    let { x, y } = { x: 5, y: "hello" };
    //let x: number;
    //let y: string;
    //let { x: a } = { x: 5, y: "hello" };  // Error, no y in target
    //let { y: b } = { x: 5, y: "hello" };  // Error, no x in target
    let { x: a, y: b } = { x: 5, y: "hello" };
    //let a: number;
    //let b: string;
}

function f3() {
    let [x, [y, [z]]] = [1, ["hello", [true]]];
    //let x: number;
    //let y: string;
    //let z: boolean;
}

function f4() {
    let { a: x, b: { a: y, b: { a: z } } } = { a: 1, b: { a: "hello", b: { a: true } } };
    //let x: number;
    //let y: string;
    //let z: boolean;
}

function f6() {
    let [x = 0, y = ""] = [1, "hello"];
    //let x: number;
    //let y: string;
}

function f7() {
    //let [x = 0, y = 1] = [1, "hello"];  // Error, initializer for y must be string
    //let x: number;
    //let y: string;
}

function f8() {
    //let [a, b, c] = [];   // Error, [] is an empty tuple
    //let [d, e, f] = [1];  // Error, [1] is a tuple
}

function f9() {
    //let [a, b] = {};                // Error, not array type
    //let [c, d] = { 0: 10, 1: 20 };  // Error, not array type
    let [e, f] = [10, 20];
}

function f10() {
    //let { a, b } = {};  // Error
    //let { a, b } = [];  // Error
}

function f11() {
    let { x: a, y: b } = { x: 10, y: "hello" };
    let { 0: a, 1: b } = { 0: 10, 1: "hello" };
    let { "<": a, ">": b } = { "<": 10, ">": "hello" };
    let { 0: a, 1: b } = [10, "hello"];
    //let a: number;
    //let b: string;
}

function f12() {
    let [a, [b, { x, y: c }] = ["abc", { x: 10, y: false }]] = [1, ["hello", { x: 5, y: true }]];
    //let a: number;
    //let b: string;
    //let x: number;
    //let c: boolean;
}

function f13() {
    let [x, y] = [1, "hello"];
    let [a, b] = [[x, y], { x: x, y: y }];
}

function f14([a = 1, [b = "hello", { x = 0, y: c = false }]]) {
    //let a: number;
    //let b: string;
    //let c: boolean;
}

module M {
    export let [a, b] = [1, 2];
}

function f15() {
    let a = "hello";
    let b = 1;
    let c = true;
    return { a, b, c };
}

function f16() {
    let { a, b, c } = f15();
}

function f17({ a = "", b = 0, c = false }) {
}

function f18() {
    let a: number;
    let b: string;
    let aa: number[];
    ({ a, b } = { a, b });
    ({ a, b } = { b, a });
    [aa[0], b] = [a, b];
    //[a, b] = [b, a];  // Error
    [a = 1, b = "abc"] = [2, "def"];
}

function f19() {
    let a, b;
    [a, b] = [1, 2];
    [a, b] = [b, a];
    ({ a, b } = { b, a });
    [[a, b] = [1, 2]] = [[2, 3]];
    let x = ([a, b] = [1, 2]);
}

function f20(v: [number, number, number]) {
    let x: number;
    let y: number;
    let z: number;
    let a0: [];
    let a1: [number];
    let a2: [number, number];
    let a3: [number, number, number];
    let [...a3] = v;
    let [x, ...a2] = v;
    let [x, y, ...a1] = v;
    //let [x, y, z, ...a0] = v;
    [...a3] = v;
    [x, ...a2] = v;
    [x, y, ...a1] = v;
    //[x, y, z, ...a0] = v;
}

function f21(v: [number, string, boolean]) {
    let x: number;
    let y: string;
    let z: boolean;
    let a0: [number, string, boolean];
    let a1: [string, boolean];
    let a2: [boolean];
    let a3: [];
    let [...a0] = v;
    let [x, ...a1] = v;
    let [x, y, ...a2] = v;
    //let [x, y, z, ...a3] = v;
    [...a0] = v;
    [x, ...a1] = v;
    [x, y, ...a2] = v;
    //[x, y, z, ...a3] = v;
}

function main() {
    f14([2, ["abc", { x: 0, y: true }]]);
    f14([2, ["abc", { x: 0 }]]);
    f14([2, ["abc", { y: false }]]);

    f17({});
    f17({ a: "hello" });
    f17({ c: true });
    f17(f15());

    print("done.");
}