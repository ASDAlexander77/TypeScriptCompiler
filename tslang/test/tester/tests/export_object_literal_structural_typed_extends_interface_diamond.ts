namespace M2 {
    export interface Left {
        left: number;
        addLeft(n: number): void;
    }
    export interface Right {
        right: number;
        addRight(n: number): void;
    }
    export interface Combined extends Left, Right {
        combined: number;
        addCombined(n: number): void;
    }

    export var rawCombined: {
        left: number;
        addLeft(n: number): void;
        right: number;
        addRight(n: number): void;
        combined: number;
        addCombined(n: number): void;
    } = {
        left: 1.0,
        addLeft(n: number) { this.left = this.left + n; },
        right: 2.0,
        addRight(n: number) { this.right = this.right + n; },
        combined: 3.0,
        addCombined(n: number) { this.combined = this.combined + n; },
    };
}
