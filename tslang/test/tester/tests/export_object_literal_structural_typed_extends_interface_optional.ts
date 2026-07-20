namespace M4 {
    export interface Base {
        base: number;
        opt?: number;
    }

    export interface Derived extends Base {
        derived: number;
    }

    export var rawPresent: {
        base: number;
        opt: number;
        derived: number;
    } = {
        base: 1.0,
        opt: 5.0,
        derived: 10.0,
    };

    export var rawMissing: {
        base: number;
        derived: number;
    } = {
        base: 2.0,
        derived: 20.0,
    };
}
