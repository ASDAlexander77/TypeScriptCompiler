// -mm=own --own-fallback=count, own only (Windows): a child stored into a field of an @atomic object,
// a global both threads read, and given a second owner by both threads at once 100000 times (own
// refuses a store of what borrows 'g.child'; with the flag it is counted). The walk at Shared.share
// does not reach it: the thread's function captures no part of it, and a global is immortal, which
// stops a walk. The store into the @atomic field is what marks it shared (the field barrier), so each
// count change is atomic and after the join its header word is its shared bit alone. Unmarked, the
// two threads lose each other's changes: extra owners are left over, or the child is freed while
// the global holds it. (Under gc there is no header word, so this is not run against gc.)
@cdecl declare function _beginthreadex(security: Opaque | null, stack: u32, start: (arg: Opaque) => u32, arg: Opaque, flags: u32, id: Opaque | null): Opaque;
@stdcall declare function WaitForSingleObject(handle: Opaque, ms: u32): u32;
@stdcall declare function CloseHandle(handle: Opaque): s32;

class Child {
    constructor(public v: number) {}
}

@atomic
class Keep {
    child: Child | null = null;
}

class Holder {
    child: Child | null = null;
}

const g = new Keep();

// what the thread runs, and what it got: as the default library's ThreadStart
class ThreadStart {
    sum: number = 0;
    constructor(public fn: () => number) {}

    run() {
        this.sum = this.fn();
    }
}

@stdcall
function entry(arg: Opaque): u32 {
    const start = <ThreadStart>arg;
    start.run();
    return 0;
}

function race(r: number): number {
    g.child = new Child(r);
    const childAt = <u64><Opaque>g.child;

    // nothing is allocated on the thread
    const onThread = new Holder();
    const start = new ThreadStart(() => {
        let sum: number = 0;
        for (let i = 0; i < 100000; i++) {
            onThread.child = g.child;
            sum += onThread.child!.v;
        }

        onThread.child = null;
        return sum;
    });

    Shared.share(start);
    const thread = _beginthreadex(null, 0, entry, <Opaque>start, 0, null);

    const onMain = new Holder();
    let sum: number = 0;
    for (let i = 0; i < 100000; i++) {
        onMain.child = g.child;
        sum += onMain.child!.v;
    }

    onMain.child = null;
    WaitForSingleObject(thread, 0xFFFFFFFF);
    CloseHandle(thread);

    // the global still holds the child, its one owner
    const word = Deref(<Ref<s64>><Opaque>(childAt - 8));
    assert(word != 0 && (word << 1) == 0, "the child's header word is its shared bit alone");
    return sum + start.sum;
}

function main() {
    let total: number = 0;
    for (let r = 0; r < 20; r++) {
        total += race(r);
    }

    print(total);
    print("done.");
}
