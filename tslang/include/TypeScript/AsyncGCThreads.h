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
// that pass runs for no other model - so `rc` and `none` reach GCThreadRegistration with the flag
// still false and do nothing, which is right: they never initialize the collector at all.
// Boehm's own GC_is_init_called cannot stand in for the flag, because the collector initializes
// itself on first use and so answers yes in programs that never meant to collect anything.

#define GC_THREADS
#include "gc.h"

namespace typescript
{
namespace asyncgc
{

inline bool &threadsEnabledFlag()
{
    static bool enabled = false;
    return enabled;
}

// Called once, from the entry point, before any coroutine can be handed to the pool.
inline void enableThreads()
{
    GC_allow_register_threads();
    threadsEnabledFlag() = true;
}

// Registered per task rather than per thread because the pool offers no thread-entry hook.
// GC_register_my_thread answers GC_DUPLICATE for a thread that is already known, and this then
// leaves the registration alone - some other task on the same thread owns it.
class ThreadRegistration
{
  public:
    ThreadRegistration() : registered(false)
    {
        if (!threadsEnabledFlag())
        {
            return;
        }

        struct GC_stack_base sb;
        if (GC_get_stack_base(&sb) == GC_SUCCESS)
        {
            registered = GC_register_my_thread(&sb) == GC_SUCCESS;
        }
    }

    ~ThreadRegistration()
    {
        if (registered)
        {
            GC_unregister_my_thread();
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
