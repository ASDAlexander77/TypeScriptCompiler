namespace L {

    // The library side of import_defaultlib_data.ts. Linked with the default library's DLL, it reads
    // data that DLL defines: PrimeHelpers.hashCollisionThreshold (core.d.ts, read by the Map
    // instantiated here), Math.PI (a const in `declare namespace Math`) and Number.EPSILON (a static
    // field of `declare class Number`). A DLL exports data only as `__imp_<name>`, so each has to be
    // read through that, or the library does not link.

    export function mapSize(n: number): number {
        const m = new Map<string, number>();
        for (let i = 0; i < n; i++) {
            m.set(`k${i}`, i);
        }

        return m.size;
    }

    export function pi(): number {
        return Math.PI;
    }

    export function epsilon(): number {
        return Number.EPSILON;
    }
}
