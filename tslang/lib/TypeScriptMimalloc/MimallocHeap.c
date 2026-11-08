// The process-heap entry points (PROCESS_HEAP_* in TypeScript/Defines.h) served by mimalloc instead
// of the Windows process heap. A Windows executable tslang links links this library ahead of
// TypeScriptAsyncRuntime, so these definitions are the ones its calls resolve to and
// TypeScriptAsyncRuntime's ProcessHeap.cpp is never pulled in.
//
// One allocator per process is still the rule (TypeScript/ProcessHeap.h): a block made by one module
// is freed by another. A program linked whole - its own code, the static default library and
// everything else it links statically - is one module, so it can own a mimalloc of its own. A
// program that loads a library tslang built (--emit=dll, a dynamic import) does not get this
// library: that library allocates from the process heap, and so does the program (exe.cpp).
//
// The same rules as ProcessHeap.h where the C functions leave room: an allocation of nothing is a
// block of its own, and a block resized to nothing is freed and nothing is returned.
//
// C, not C++: it pulls in no C++ runtime library, whose debug/release checks (LNK2038) a `--di`
// program linking the release-CRT runtime libraries of a tslang package would trip on.

#ifdef _WIN32

#include <stddef.h>

#include <mimalloc.h>

// mimalloc asks for the large-page privilege through advapi32 (win_enable_large_os_pages), and
// neither tslang nor test-runner links advapi32 into a program otherwise
#pragma comment(lib, "advapi32.lib")

void *__tslang_heap_malloc(size_t size)
{
    return mi_malloc(size ? size : 1);
}

void *__tslang_heap_calloc(size_t count, size_t size)
{
    if (count == 0 || size == 0)
    {
        return mi_calloc(1, 1);
    }

    // mi_calloc refuses a count * size that overflows
    return mi_calloc(count, size);
}

void *__tslang_heap_realloc(void *ptr, size_t size)
{
    if (ptr && size == 0)
    {
        mi_free(ptr);
        return NULL;
    }

    return mi_realloc(ptr, size);
}

void __tslang_heap_free(void *ptr)
{
    mi_free(ptr);
}

void *__tslang_heap_aligned_alloc(size_t alignment, size_t size)
{
    return mi_malloc_aligned(size ? size : 1, alignment);
}

#endif // _WIN32
