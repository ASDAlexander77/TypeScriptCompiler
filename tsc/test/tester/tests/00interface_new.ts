interface ClockConstructor {
    new(h: number, m: number): ClockInterface;
}

interface ClockInterface {
    tick(): void;
    h: number;
    m: number;
}

class Clock implements ClockConstructor {
    constructor(public h: number, public m: number) { print(`Call ctor : ${h}, ${m}`); }
    tick() {
        print("beep beep");
    }
};

function main() {
    const clockInst: ClockConstructor = new Clock(20, 30);
    const newInst = new clockInst(1, 2);
    newInst.tick();

    assert(newInst.h == 1);
    assert(newInst.m == 2);

    print("done.");
}