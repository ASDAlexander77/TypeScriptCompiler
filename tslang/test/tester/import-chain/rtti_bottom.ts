// The class whose .rtti the program reads through two imports; see rtti_skipped_import.ts.

export class Profiler {
    reads = 0;

    read(): boolean {
        this.reads++;
        return this.reads > 1;
    }
}
