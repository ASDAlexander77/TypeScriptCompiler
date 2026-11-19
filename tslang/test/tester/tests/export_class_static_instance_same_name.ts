namespace M {

    // A static and an instance member of one name, in the exporting module: the importer reaches the
    // static one by the class and the instance one by an object, each the function this module made.
    export class Range {
        constructor(public low: number, public high: number) {}

        public contains(x: number): boolean {
            return x >= this.low && x <= this.high;
        }

        public static contains(range: Range, x: number): boolean {
            return x > range.low && x < range.high;
        }

        // (an imported static getter is not called yet, twin or not)
        public get kind(): string {
            return "closed";
        }
    }
}
