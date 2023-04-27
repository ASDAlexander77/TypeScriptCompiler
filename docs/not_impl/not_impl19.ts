interface ClockConstructor {
    new(hour: number, minute: number): ClockInterface;
}

interface ClockInterface {
    tick(): void;
}

function main() {
    const ClockInst: ClockConstructor = class Clock implements ClockInterface {
        constructor(h: number, m: number) { }
        tick() {
            print("beep beep");
        }
    };

    let clock = new ClockInst(12, 17);
    clock.tick();
}