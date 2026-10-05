// The async runtime's side of the collector, in an object file of its own - see AsyncGCThreads.h.
//
// Shared by TypeScriptRuntime/AsyncGCThreads.cpp (the JIT's DLL) and
// TypeScriptAsyncRuntime/AsyncGCThreads.cpp (the static library ahead-of-time builds link). In the
// static library nothing but GC_enable_threads and __tslang_gc_enter refer to this object, and
// only a `gc` program or library calls those, so only they pull it in - and with it the
// collector's thread API.

#define GC_THREADS
#include "gc.h"

#include "TypeScript/AsyncGCThreads.h"

#include <mutex>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace
{

bool registerThisThread()
{
    struct GC_stack_base sb;
    return GC_get_stack_base(&sb) == GC_SUCCESS && GC_register_my_thread(&sb) == GC_SUCCESS;
}

void unregisterThisThread()
{
    GC_unregister_my_thread();
}

// A thread that came into a gc shared library through one of its exported functions, from a host
// that is not a tslang program (an Android app's JNI threads, a C program's): checked once, and
// registered then if the collector did not know it, until the thread ends.
struct CallingThread
{
    bool checked = false;
    // registered by __tslang_gc_enter, so unregistered here, at thread exit
    bool registeredHere = false;

    ~CallingThread()
    {
        if (registeredHere)
        {
            GC_unregister_my_thread();
        }
    }
};

thread_local CallingThread callingThread;

std::once_flag threadsEnabled;

} // namespace

extern "C" void GC_enable_threads();

// Called first thing by every exported function of a `-mm=gc` shared library (the GC pass injects
// it). The collector stops and scans only the threads registered with it, and aborts a collection
// that a thread it does not know started ("Collecting from unknown thread"): a tslang program
// registers the threads it makes, but a library's host makes its own. GC_init registered the one
// it ran on; any other is registered on its first call.
//
// GC_enable_threads runs once here whatever ran before: GC_register_my_thread refuses every
// thread until GC_allow_register_threads was called ("Threads explicit registering is not
// previously enabled"), and the collector may well be initialized already without it - it
// initializes itself on first use, so a library with no top-level code (no GC_init injected at
// load) can find it running. Calling it again where GC_init's injection already did is harmless.
extern "C" void __tslang_gc_enter()
{
    auto &thread = callingThread;
    if (thread.checked)
    {
        return;
    }

    std::call_once(threadsEnabled, [] {
        if (!GC_is_init_called())
        {
            GC_init();
        }

        GC_enable_threads();
    });

    thread.checked = true;
    if (!GC_thread_is_registered())
    {
        thread.registeredHere = registerThisThread();
    }
}

// Called once from the entry point: the GC pass injects the call beside GC_init, so it happens
// only in a `gc` build. When linked into the shared TypeScriptRuntime, this symbol is also
// exported under this name in TypeScriptRuntime.def for the JIT to resolve.
extern "C" void GC_enable_threads()
{
    GC_allow_register_threads();

    auto &hooks = typescript::asyncgc::threadHooks();
    hooks.registerThread = &registerThisThread;
    hooks.unregisterThread = &unregisterThisThread;
}
