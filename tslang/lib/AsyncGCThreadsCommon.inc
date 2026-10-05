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

#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <winternl.h>
#else
#include <pthread.h>
#endif

namespace
{

// Whether this thread is inside a DllMain - loading or unloading some DLL - where it holds the
// loader lock, and a thread it starts cannot run until the lock is released. The lock is the
// PEB's LoaderLock, which winternl.h leaves out of its PEB; its offset has stayed the same since
// Windows NT 4. Off Windows there is no such lock: a shared object's constructors run under
// dlopen's lock, which a new thread does not need to start.
bool holdsLoaderLock()
{
#ifdef _WIN32
#ifdef _WIN64
    constexpr size_t loaderLockOffset = 0x110;
#else
    constexpr size_t loaderLockOffset = 0xA0;
#endif
    auto peb = reinterpret_cast<const char *>(NtCurrentTeb()->ProcessEnvironmentBlock);
    auto loaderLock = *reinterpret_cast<RTL_CRITICAL_SECTION *const *>(peb + loaderLockOffset);
    return loaderLock != nullptr &&
           reinterpret_cast<ULONG_PTR>(loaderLock->OwningThread) == static_cast<ULONG_PTR>(GetCurrentThreadId());
#else
    return false;
#endif
}

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
// that is not a tslang program (an Android app's JNI threads, a C program's): checked once threads
// are enabled, and registered then if the collector did not know it, until the thread ends.
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

std::once_flag collectorInitialized;
std::once_flag threadsEnabledOnce;
std::atomic<bool> threadsEnabled{false};

} // namespace

extern "C" void GC_enable_threads();

namespace
{

// Runs GC_enable_threads once, unless now is no time for it: false then, and a later call does.
//
// Not inside a DllMain: GC_allow_register_threads starts the collector's markers and waits for
// them to come up, which they cannot do until the loader lock is released, so the load never
// finished. An exported function runs there when the library calls one while it loads - a static
// constructor of an exported class, top-level code calling an export (the default library hung
// every JIT run). Nor does such a call wait for another thread that is enabling them: that one
// waits for the markers, and they for this thread's loader lock. The markers running already
// (GC_get_parallel) would not tell it either: the count is set before they are waited for.
bool enableThreads()
{
    if (threadsEnabled.load(std::memory_order_acquire))
    {
        return true;
    }

    if (holdsLoaderLock())
    {
        return false;
    }

    std::call_once(threadsEnabledOnce, [] {
        GC_enable_threads();
        threadsEnabled.store(true, std::memory_order_release);
    });

    return true;
}

} // namespace

// Called first thing by every exported function of a `-mm=gc` shared library (the GC pass injects
// it). The collector stops and scans only the threads registered with it, and aborts a collection
// that a thread it does not know started ("Collecting from unknown thread"): a tslang program
// registers the threads it makes, but a library's host makes its own. GC_init registered the one
// it ran on; any other is registered on its first call.
//
// The collector may well be initialized already: it initializes itself on first use, so a library
// with no top-level code (no GC_init injected at load) can find it running.
//
// The first call that may enables threads (enableThreads), whichever thread makes it - before it
// goes on, and before any other thread can come in:
// - GC_register_my_thread refuses every thread until GC_allow_register_threads was called
//   ("Threads explicit registering is not previously enabled");
// - GC_allow_register_threads makes the allocator take its lock only after it started the markers,
//   so a newcomer enabling threads raced a thread that was already allocating (#523);
// - the library's own async pool registers its threads through hooks only the library's
//   GC_enable_threads sets - each module has its own copy of the scheduler - so they must be set
//   even when every caller is registered already (#522).
// While the library loads, the call cannot enable them (see enableThreads) and the thread goes on
// as it is: the one loading is the one GC_init registered, or one its host registered, and no other
// can call in before the load finishes. Its next call, after the load, enables them.
extern "C" void __tslang_gc_enter()
{
    auto &thread = callingThread;
    if (thread.checked)
    {
        return;
    }

    std::call_once(collectorInitialized, [] {
        if (!GC_is_init_called())
        {
            GC_init();
        }
    });

    if (!enableThreads())
    {
        return;
    }

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
