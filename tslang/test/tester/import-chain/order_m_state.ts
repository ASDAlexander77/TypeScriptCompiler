// Holds a Profiler | null; see order_program.ts.
import { Profiler } from './order_z_profiler'

export class State {
    protected profiler: Profiler | null = null;

    start() {
        this.profiler = new Profiler();
    }
}
