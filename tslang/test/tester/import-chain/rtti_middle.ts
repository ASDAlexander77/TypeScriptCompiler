// Holds a Profiler | null; see rtti_skipped_import.ts.
import { Profiler } from './rtti_bottom'

export class State {
    protected profiler: Profiler | null = null;

    start() {
        this.profiler = new Profiler();
    }
}
