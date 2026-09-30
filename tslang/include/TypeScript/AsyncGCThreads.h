#ifndef MLIR_TYPESCRIPT_ASYNCGCTHREADS_H_
#define MLIR_TYPESCRIPT_ASYNCGCTHREADS_H_

// Shared by the two copies of the async runtime - the one inside TypeScriptRuntime.dll that the
// JIT resolves against, and TypeScriptAsyncRuntime.lib that an ahead-of-time build links.
//
// A coroutine is resumed on one of the runtime's pool threads, and under `-mm=gc` its frame and
// everything its body builds come from the collector: the frame is handed back with GC_free at the
// end of the resume. Two things follow from that, and neither was being done.
//
// The one that crashed: Boehm does not lock its allocator until it is told there is more than one
// thread. `GC_need_to_lock` starts FALSE and LOCK()/UNLOCK() expand to nothing, so a worker and
// the awaiting thread walked the same free lists at once with no lock at all. That is what made
// `-mm=gc` fault on a long chain of awaits - about 1 run in 4 at 200k frames, never at 50k, never
// under `rc` or `none`, whose frames go to the CRT heap. Nothing to do with collection: a 1 GB
// initial heap, which leaves nothing to collect, changed nothing, and running the tasks inline
// made it stop. `GC_allow_register_threads` is what sets the flag.
//
// The one that would have come next: a thread the collector has never heard of is not suspended
// during a collection and its stack is not scanned, so a frame held only in that worker's
// registers can be freed underneath it. Hence the registration below.
//
// Only a `gc` build calls GC_enable_threads - the GC pass injects the call beside GC_init, and
// that pass runs for no other model. Boehm's own GC_is_init_called cannot stand in for that,
// because the collector initializes itself on first use and so answers yes in programs that never
// meant to collect anything.
//
// Nothing here touches the collector itself: GC_enable_threads, and the registration it installs
// as the hooks below, are in AsyncGCThreadsCommon.inc - an object file of its own, which only the
// call to GC_enable_threads pulls in. Every async program links the scheduler; had the scheduler
// called GC_register_my_thread itself, every `rc`, `none` and `own` program that awaits anything
// would need the collector to link, and they do not link it.

namespace typescript
{
namespace asyncgc
{

// Left null unless GC_enable_threads ran, which it does once, from the entry point, before any
// coroutine can be handed to the pool.
struct ThreadHooks
{
    // true when this call registered the thread (and so has to unregister it)
    bool (*registerThread)() = nullptr;
    void (*unregisterThread)() = nullptr;
};

inline ThreadHooks &threadHooks()
{
    static ThreadHooks hooks;
    return hooks;
}

// Registered per task rather than per thread because the pool offers no thread-entry hook.
// GC_register_my_thread answers GC_DUPLICATE for a thread that is already known, and this then
// leaves the registration alone - some other task on the same thread owns it.
class ThreadRegistration
{
  public:
    ThreadRegistration() : registered(false)
    {
        auto registerThread = threadHooks().registerThread;
        if (registerThread != nullptr)
        {
            registered = registerThread();
        }
    }

    ~ThreadRegistration()
    {
        if (registered)
        {
            threadHooks().unregisterThread();
        }
    }

    ThreadRegistration(const ThreadRegistration &) = delete;
    ThreadRegistration &operator=(const ThreadRegistration &) = delete;

  private:
    bool registered;
};

} // namespace asyncgc
} // namespace typescript

#endif // MLIR_TYPESCRIPT_ASYNCGCTHREADS_H_
