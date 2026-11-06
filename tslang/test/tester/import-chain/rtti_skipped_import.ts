// In a discovery module an import discovered already is not generated again, and the globals it
// registered are put back instead. Here rtti_top is generated into the program's discovery module,
// while rtti_middle and rtti_bottom, discovered inside rtti_top's discovery, are skipped. Testing
// a Profiler | null for truth reads Profiler's .rtti, a global rtti_bottom registered: without it
// the program failed with "Class member '.rtti' can't be found".
import { Saves } from './rtti_top'

class Simulation extends Saves {
    reports = 0;

    readProfile() {
        const profiler = this.profiler;
        if (profiler && profiler.read()) {
            this.reports++;
        }
    }
}

const sim = new Simulation();
sim.readProfile();
assert(sim.reports == 0);
sim.start();
sim.readProfile();
sim.readProfile();
assert(sim.reports == 1);

print("done.");
