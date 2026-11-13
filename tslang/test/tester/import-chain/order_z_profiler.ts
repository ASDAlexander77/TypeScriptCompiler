// The class the others name; its module sorts last by name. See order_program.ts.

export class Profiler {
    reads = 0;

    read(): boolean {
        this.reads++;
        return this.reads > 1;
    }
}
