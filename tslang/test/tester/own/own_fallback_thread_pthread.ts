// -mm=own --own-fallback=count across threads (POSIX; own_fallback_thread.ts is the Windows
// twin). A node kept by a holder on the main thread and captured by a closure another thread runs
// has two owners, which own refuses (the closure captures what borrows 'keep.node'); with the flag it
// is counted. Its child has one owner, the node, and both threads give it a second one 100000 times
// over (a holder's field overwritten with it: one count taken, the one before given back), at once.
// The thread is started as Thread.spawn does, its start marked shared first (Shared.share): the walk
// sets the shared bit of every block it reaches - the start, the closure's box, the node, the child -
// so those counts change atomically. Counted plainly, two threads lose each other's changes and the
// child is freed while held, or never.
@cdecl declare function pthread_create(thread: Opaque, attributes: Opaque | null, start: (arg: Opaque) => Opaque, arg: Opaque): s32;
declare function pthread_join(thread: u64, result: Opaque | null): s32;

class Child {
    constructor(public v: number) {}
}

class Node {
    constructor(public v: number, public child: Child) {}
}

class Holder {
    node: Node | null = null;
    child: Child | null = null;
}

// what the thread runs, and what it got: as the default library's ThreadStart
class ThreadStart {
    sum: number = 0;
    constructor(public fn: () => number) {}

    run() {
        this.sum = this.fn();
    }
}

function entry(arg: Opaque): Opaque {
    const start = <ThreadStart>arg;
    start.run();
    return arg;
}

function race(r: number): number {
    const keep = new Holder();
    const n = new Node(r, new Child(r + 1));
    keep.node = n;

    // nothing is allocated on the thread: under gc it is not one the collector knows
    const onThread = new Holder();
    const start = new ThreadStart(() => {
        let sum: number = 0;
        for (let i = 0; i < 100000; i++) {
            onThread.child = n.child;
            sum += n.v + onThread.child!.v;
        }

        onThread.child = null;
        return sum;
    });

    Shared.share(start);
    let thread: u64 = 0;
    assert(pthread_create(<Opaque>Ref(thread), null, entry, <Opaque>start) == 0, "pthread_create");

    const onMain = new Holder();
    let sum: number = 0;
    for (let i = 0; i < 100000; i++) {
        onMain.child = keep.node!.child;
        sum += keep.node!.v + onMain.child!.v;
    }

    onMain.child = null;
    keep.node = null;

    assert(pthread_join(thread, null) == 0, "pthread_join");
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
