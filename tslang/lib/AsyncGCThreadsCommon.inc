// The async runtime's side of the collector, in an object file of its own - see AsyncGCThreads.h.
//
// Shared by TypeScriptRuntime/AsyncGCThreads.cpp (the JIT's DLL) and
// TypeScriptAsyncRuntime/AsyncGCThreads.cpp (the static library ahead-of-time builds link). In the
// static library nothing but GC_enable_threads refers to this object, and only a `gc` program
// calls that, so only a `gc` program pulls it in - and with it the collector's thread API.

#define GC_THREADS
#include "gc.h"

#include "TypeScript/AsyncGCThreads.h"

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

} // namespace

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
