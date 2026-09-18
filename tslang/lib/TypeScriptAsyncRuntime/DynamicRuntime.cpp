//===- DynamicRuntime.cpp - Shared library loading for AOT executables -----===//
//
// A program that imports a shared library loads it from a global constructor and then resolves
// each imported symbol by name, through the two functions below. Under the JIT, TypeScriptRuntime
// supplies them by wrapping llvm::sys::DynamicLibrary (see TypeScriptRuntime/DynamicRuntime.cpp).
// An executable used to call LLVMLoadLibraryPermanently and LLVMSearchForAddressOfSymbol from
// LLVMSupport instead. These are here so that it does not have to link any LLVM library at all.
//
// They behave as LLVM's do for what an executable asks of them: a library is loaded once and
// kept for the life of the process, and a symbol is looked up in the loaded libraries in the
// order they were loaded.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{

// Function-local statics: the library is loaded from a global constructor, which can run before
// this file's own namespace-scope objects are initialized.
std::mutex &handlesMutex()
{
    static std::mutex mu;
    return mu;
}

std::vector<void *> &loadedHandles()
{
    static std::vector<void *> handles;
    return handles;
}

void *openLibrary(const char *fileName)
{
#ifdef _WIN32
    auto size = MultiByteToWideChar(CP_UTF8, 0, fileName, -1, nullptr, 0);
    if (size <= 0)
    {
        return nullptr;
    }

    std::wstring wideFileName(size, L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, fileName, -1, wideFileName.data(), size) <= 0)
    {
        return nullptr;
    }

    return reinterpret_cast<void *>(LoadLibraryW(wideFileName.c_str()));
#else
    return dlopen(fileName, RTLD_LAZY | RTLD_GLOBAL);
#endif
}

void *findSymbol(void *handle, const char *symbolName)
{
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbolName));
#else
    return dlsym(handle, symbolName);
#endif
}

} // namespace

// 0 when the library is loaded (or already was), 1 when it cannot be.
extern "C" int tslang_load_library_permanently(const char *fileName)
{
    if (!fileName)
    {
        // LLVM reads a null name as "the process itself"; nothing the compiler emits asks for that
        return 1;
    }

    auto handle = openLibrary(fileName);
    if (!handle)
    {
        return 1;
    }

    std::lock_guard<std::mutex> lock(handlesMutex());
    auto &handles = loadedHandles();
    for (auto loaded : handles)
    {
        if (loaded == handle)
        {
            // the loader counts every open, and a library loaded for good is never closed, so
            // opening it again changes nothing
            return 0;
        }
    }

    handles.push_back(handle);
    return 0;
}

extern "C" void *tslang_search_for_address_of_symbol(const char *symbolName)
{
    std::lock_guard<std::mutex> lock(handlesMutex());
    for (auto handle : loadedHandles())
    {
        if (auto address = findSymbol(handle, symbolName))
        {
            return address;
        }
    }

    return nullptr;
}
