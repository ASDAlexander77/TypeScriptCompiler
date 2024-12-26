// @strict-null false
type Action = () => void;

let glb1 = 0;
let x = 0;
let action: Action;
let sum = 0;
let tot: string = null;

class Testrec {
    str: string;
    num: number;
    bool: boolean;
    str2: string;
}

function pause(ms: number): void {}

namespace control {
    export function runInBackground(a: () => void): void {
        a();
    }
}

function testRefLocals(): void {
    print("start test ref locals");
    let s = "";
    for (let i of [3, 2, 1]) {
        let copy = i;
        control.runInBackground(() => {
            pause(10 * i);
            copy = copy + 10;
        });
        control.runInBackground(() => {
            pause(20 * i);
            s = s + copy;
        });
    }
    pause(200);
    //assert(s == "111213", "reflocals");
    assert(s == "131211", "reflocals");
}

function byRefParam_0(p: number): void {
    control.runInBackground(() => {
        pause(1);
        sum = sum + p;
    });
    p = p + 1;
}

function byRefParam_2(pxx: number): void {
    pxx = pxx + 1;
    control.runInBackground(() => {
        pause(1);
        sum = sum + pxx;
    });
}

function testByRefParams(): void {
    print("testByRefParams");
    refparamWrite("a" + "b");
    refparamWrite2(new Testrec());
    //refparamWrite3(new Testrec());
    sum = 0;
    let x = 1;
    control.runInBackground(() => {
        pause(1);
        sum = sum + x;
    });
    x = 2;
    byRefParam_0(4);
    byRefParam_2(10);
    pause(330);
    //assert(sum == 18, "by ref");
    assert(sum == 16, "by ref");
    sum = 0;
    print("byref done");
}

function refparamWrite(s: string): void {
    s = s + "c";
    assert(s == "abc", "abc");
}

function refparamWrite2(testrec: Testrec): void {
    testrec = new Testrec();
    // TODO:
    //assert(testrec.bool === undefined, "rw2f");
    assert(testrec.bool == false, "rw2");
}

function refparamWrite3(testrecX: Testrec): void {
    control.runInBackground(() => {
        pause(1);
        assert(testrecX.str == "foo", "ff");
        testrecX.str = testrecX.str + "x";
    });
    testrecX = new Testrec();
    testrecX.str = "foo";
    pause(130);
    assert(testrecX.str == "foox", "ff2");
}

function main() {
    testRefLocals();
    testByRefParams();
    print("done.");
}
