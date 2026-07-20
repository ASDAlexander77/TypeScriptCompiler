namespace M3 {
    export interface A {
        a: number;
        addA(n: number): void;
    }
    export interface B {
        b: number;
        addB(n: number): void;
    }
    export interface C {
        c: number;
        addC(n: number): void;
    }
    export interface Combined extends A, B, C {
        combined: number;
        addCombined(n: number): void;
    }

    export var rawCombined: {
        a: number;
        addA(n: number): void;
        b: number;
        addB(n: number): void;
        c: number;
        addC(n: number): void;
        combined: number;
        addCombined(n: number): void;
    } = {
        a: 1.0,
        addA(n: number) { this.a = this.a + n; },
        b: 2.0,
        addB(n: number) { this.b = this.b + n; },
        c: 3.0,
        addC(n: number) { this.c = this.c + n; },
        combined: 4.0,
        addCombined(n: number) { this.combined = this.combined + n; },
    };
}
