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

        public get kind(): string {
            return "closed";
        }

        public static get kind(): string {
            return "Range";
        }

        // a static getter and setter alone: the importer calls them through the addresses the
        // library gives when it loads
        private static count = 0;

        public static get made(): number {
            return Range.count;
        }

        public static set made(value: number) {
            Range.count = value;
        }
    }
}
