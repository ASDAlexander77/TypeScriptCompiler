// A class an object literal holds in another module (import_record_stamp.ts), as DefaultLib's Date is
// held by `JSON.stringify({ when: new Date(0) })`.
export class Stamp {
    constructor(public at: number) {}

    toString() {
        return "stamp " + this.at;
    }
}
