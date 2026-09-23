namespace M {

    // Thrown only from this module: the importer catches it without ever throwing it itself.
    export class Failure {
        code: number;

        constructor(code: number) {
            this.code = code;
        }
    }

    export function fail(code: number) {
        throw new Failure(code);
    }
}
