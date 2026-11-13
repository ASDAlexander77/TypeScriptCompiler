// rtti_skipped_import.ts with its modules named the other way round. Under -shared the three are
// one library, which declares each module from its own __decls_<module>, and they name each other's
// classes: State holds a Profiler. They were declared in the order the library's symbol table lists
// them - by name in a DLL (order_a_saves, order_m_state, order_z_profiler: State before Profiler), as
// the linker laid them out in a shared object - and the first that named what was not declared yet
// failed the import: "can't find type by name: Profiler". Now one that fails is read again after
// the others.
import { Saves } from './order_a_saves'

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
