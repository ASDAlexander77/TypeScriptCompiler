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

function pause(ms: number): void { }

namespace control {
    export function runInBackground(a: () => void): void {
        a();
    }
}

function inBg() {
    let k = 7;
    let q = 14;
    let rec = new Testrec();
    glb1 = 0;
    control.runInBackground(() => {
        glb1 = glb1 + 10 + (q - k);
        rec.str = "foo";
    });
    control.runInBackground(() => {
        glb1 = glb1 + 1;
    });
    pause(50);
    assert(glb1 == 18, "inbg0");
    assert(rec.str == "foo", "inbg1");
    glb1 = 0;
}

function runTwice(fn: Action): void {
    print("r2 start");
    fn();
    fn();
    print("r2 stop");
}

function iter(max: number, fn: (v: number) => void) {
    for (let i = 0; i < max; ++i) {
        fn(i);
    }
}

function testIter() {
    x = 0;
    iter(10, (v) => {
        x = x + (v + 1);
    });
    assert(x == 55, "55");
    x = 0;
}

function testAction(p: number): void {
    print("testActionStart");
    let s = "hello" + "1";
    let coll = [] as number[];
    let p2 = p * 2;
    x = 42;
    runTwice(() => {
        x = x + p + p2;
        coll.push(x);
        print(s + x);
    });
    assert(x == 42 + p * 6, "run2");
    assert(coll.length == 2, "run2");
    x = 0;
    print("testActionDone");
}

function add7() {
    sum = sum + 7;
}

function testFunDecl() {
    print("testFunDecl");
    let x = 12;
    sum = 0;
    function addX() {
        sum = sum + x;
    }
    function add10() {
        sum = sum + 10;
    }
    runTwice(addX);
    assert(sum == 24, "cap");
    print("testAdd10");
    runTwice(add10);
    print("end-testAdd10");
    assert(sum == 44, "nocap");
    runTwice(add7);
    assert(sum == 44 + 14, "glb");
    addX();
    add10();
    assert(sum == 44 + 14 + x + 10, "direct");
    sum = 0;
}

function saveAction(fn: Action): void {
    action = fn;
}

function saveGlobalAction(): void {
    let s = "foo" + "42";
    tot = "";
    saveAction(() => {
        tot = tot + s;
    });
}

function testActionSave(): void {
    saveGlobalAction();
    print("saveAct");
    runTwice(action);
    print("saveActDONE");
    print(tot);
    assert(tot == "foo42foo42", "");
    tot = "";
    action = null;
}

function testLoopScope() {
    for (let i = 0; i < 3; ++i) {
        let val: number;
        // TODO:
        //assert(val === undefined, "loopscope");
        val = i;
    }
}

function main() {
    inBg();
    testAction(1);
    testAction(7);
    testIter();
    testActionSave();
    testFunDecl();
    testLoopScope();
    print("done.");
}
