namespace M {

    // An interface with generic methods (erased to `any` slots) and a class implementing it, both in
    // the exporting module: the importer calls through the interface with its own type arguments,
    // implements the interface itself, and gives its own objects to the exported function.

    export interface Mapper {
        map<T>(x: T): T;
        tag<T>(x: T, label: string): string;
    }

    export class Id implements Mapper {
        map<T>(x: T): T {
            return x;
        }

        tag<T>(x: T, label: string): string {
            return "id:" + label;
        }
    }

    export function useMapper(m: Mapper): number {
        return m.map<number>(40) + 2;
    }
}
