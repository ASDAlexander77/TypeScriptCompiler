#ifndef TYPESCRIPT_UNCAUGHTSTACKTRACE_H_
#define TYPESCRIPT_UNCAUGHTSTACKTRACE_H_

// Where an exception nothing caught was thrown from, said under "uncaught exception: ..." by the JIT
// (tslang/jit.cpp) and an executable's runtime (TypeScriptAsyncRuntime/UncaughtException.cpp):
//
//     error: uncaught exception: Failure: not caught
//         at fail (I:\src\program.ts:7)
//         at main (I:\src\program.ts:12)
//
// Always, not only with --di: the frames are walked only on the way out of a program that failed (on
// Windows a throw also copies its context, see noteThrow). What a
// frame can say depends on the build. A Windows executable without --di has no PDB, so its frames are
// `program.exe+0x1a2b` and a line says to compile with --di; with --di dbghelp reads the function and the
// line from the PDB. A Linux executable keeps its symbol table, so its frames have names, and with --di
// `addr2line` (when it is installed) adds the line. JIT'd code is named by the JIT (an UncaughtFrameNamer).
//
// The frames are those of the throw: on Windows the context of the exception record, read in the
// unhandled-exception filter, before anything is unwound - or, when that is a `finally` throwing it again,
// the context of the first throw (thrownFrom); on Linux std::terminate() is called where the throw found
// no catch (a `finally` is a cleanup there, which the search passes), so the throw is still on the stack. The runtime's own frames are left out - the
// throw (RaiseException, _CxxThrowException, __cxa_throw), the C runtime's startup after main and, in the
// JIT, tslang's own frames around the program.

#include "TypeScript/UncaughtException.h"

#ifdef TSLANG_DESCRIBES_UNCAUGHT

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <dbghelp.h> // the types; dbghelp.dll is loaded only when a trace is said
#else
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unwind.h>
#endif

namespace typescript
{

constexpr int uncaughtStackMaxFrames = 64;

// What a frame is called: its function, and its source line when debug info has one
struct UncaughtFrameName
{
    char function[256];
    char file[512];
    unsigned line;
};

// Names a frame of code no module holds (the JIT's); false for one it does not know
using UncaughtFrameNamer = bool (*)(uintptr_t pc, UncaughtFrameName &name);

namespace uncaught_stack
{

enum class FrameKind
{
    Program,
    // the C/C++ runtime and the system: never said
    System,
    // the JIT: tslang itself, around the program
    Host,
};

inline bool startsWith(const char *text, const char *prefix)
{
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

#ifdef _WIN32

inline bool startsWithNoCase(const char *text, const char *prefix)
{
    return _strnicmp(text, prefix, strlen(prefix)) == 0;
}

inline const char *baseName(const char *path)
{
    auto *name = path;
    for (auto *at = path; *at; at++)
    {
        if (*at == '\\' || *at == '/')
        {
            name = at + 1;
        }
    }

    return name;
}

struct DbgHelp
{
    HANDLE process;
    decltype(&SymFromAddr) fromAddr;
    decltype(&SymGetLineFromAddr64) lineFromAddr;
    decltype(&SymGetModuleInfo64) moduleInfo;
#ifdef _M_IX86
    decltype(&StackWalk64) stackWalk;
    decltype(&SymFunctionTableAccess64) functionTableAccess;
    decltype(&SymGetModuleBase64) moduleBase;
#endif
};

// dbghelp.dll from System32, initialized once; null when it can't be had
inline DbgHelp *dbgHelp()
{
    static DbgHelp dbg;
    static bool tried;
    if (tried)
    {
        return dbg.fromAddr ? &dbg : nullptr;
    }

    tried = true;
    auto library = LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library)
    {
        return nullptr;
    }

    auto setOptions = reinterpret_cast<decltype(&SymSetOptions)>(GetProcAddress(library, "SymSetOptions"));
    auto initialize = reinterpret_cast<decltype(&SymInitialize)>(GetProcAddress(library, "SymInitialize"));
    auto fromAddr = reinterpret_cast<decltype(&SymFromAddr)>(GetProcAddress(library, "SymFromAddr"));
    dbg.lineFromAddr = reinterpret_cast<decltype(&SymGetLineFromAddr64)>(GetProcAddress(library, "SymGetLineFromAddr64"));
    dbg.moduleInfo = reinterpret_cast<decltype(&SymGetModuleInfo64)>(GetProcAddress(library, "SymGetModuleInfo64"));
#ifdef _M_IX86
    dbg.stackWalk = reinterpret_cast<decltype(&StackWalk64)>(GetProcAddress(library, "StackWalk64"));
    dbg.functionTableAccess =
        reinterpret_cast<decltype(&SymFunctionTableAccess64)>(GetProcAddress(library, "SymFunctionTableAccess64"));
    dbg.moduleBase = reinterpret_cast<decltype(&SymGetModuleBase64)>(GetProcAddress(library, "SymGetModuleBase64"));
#endif
    if (!setOptions || !initialize || !fromAddr || !dbg.lineFromAddr || !dbg.moduleInfo)
    {
        return nullptr;
    }

    dbg.process = GetCurrentProcess();
    setOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_FAIL_CRITICAL_ERRORS |
               SYMOPT_NO_PROMPTS);

    // A PDB is looked for where the image says it is, then beside the executable - not on a symbol server
    // _NT_SYMBOL_PATH may name, which a failing program would wait on. Already initialized (LLVM's crash
    // handler in tslang) is as good.
    char searchPath[MAX_PATH] = "";
    auto length = GetModuleFileNameA(nullptr, searchPath, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
    {
        const_cast<char *>(baseName(searchPath))[0] = '\0';
    }

    initialize(dbg.process, searchPath[0] ? searchPath : nullptr, TRUE);
    dbg.fromAddr = fromAddr;
    return &dbg;
}

} // namespace uncaught_stack
} // namespace typescript

// vcruntime's: the exception whose catch block runs on this thread, and the context it was thrown in
extern "C" void **__cdecl __current_exception();
extern "C" void **__cdecl __current_exception_context();

namespace typescript
{
namespace uncaught_stack
{

constexpr DWORD cxxExceptionCode = 0xE06D7363; // 'msc' | 0xE0000000, a throw

// The object a throw's record throws: null for `throw;` as the JIT raises it (the exception being caught)
inline ULONG_PTR thrownObject(const EXCEPTION_RECORD *record)
{
    return record->NumberParameters >= 2 ? record->ExceptionInformation[1] : 0;
}

// The object of the exception whose catch block runs on this thread, or 0
inline ULONG_PTR caughtObject()
{
    __try
    {
        auto *current = static_cast<EXCEPTION_RECORD *>(*__current_exception());
        return current ? thrownObject(current) : 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return 0;
    }
}

// A throw again of the exception a catch block caught: `throw;`, which a `finally` and a scope that releases
// what it holds lower to
inline bool isRethrow(const EXCEPTION_RECORD *record)
{
    auto object = thrownObject(record);
    return !object || object == caughtObject();
}

// The last throw on this thread that was not a rethrow: its record and context, copied as it is raised
struct ThrowSite
{
    EXCEPTION_RECORD record;
    CONTEXT context;
    EXCEPTION_POINTERS pointers;
    bool set;
};

inline ThrowSite &throwSite()
{
    static thread_local ThrowSite site;
    return site;
}

// A vectored handler: it sees every throw before any frame's handler does, and a rethrow does not replace
// the throw it throws again. The copy is all it costs a throw.
inline LONG WINAPI noteThrow(EXCEPTION_POINTERS *pointers)
{
    auto *record = pointers->ExceptionRecord;
    if (record->ExceptionCode == cxxExceptionCode && !isRethrow(record))
    {
        auto &site = throwSite();
        site.record = *record;
        site.record.ExceptionRecord = nullptr;
        site.context = *pointers->ContextRecord;
        site.set = true;
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

inline void noteThrows()
{
    AddVectoredExceptionHandler(0, noteThrow);
}

// Where the exception that reached the filter was thrown. A `finally` catches everything and throws it
// again: what reaches the filter is then that rethrow, from a catch block that runs where vcruntime has
// unwound to the function with the `finally`, so its frames end there. The first throw's frames are still on
// the stack, under the catch blocks, and its record (noteThrow) says what it threw - for the JIT's `throw;`,
// which raises no object, too.
inline EXCEPTION_POINTERS *thrownFrom(EXCEPTION_POINTERS *pointers)
{
    auto *record = pointers->ExceptionRecord;
    auto &site = throwSite();
    if (record->ExceptionCode != cxxExceptionCode || !site.set || !isRethrow(record))
    {
        return pointers;
    }

    // a throw caught and done with inside the catch block is not the one thrown again
    auto object = thrownObject(record) ? thrownObject(record) : caughtObject();
    if (object && object != thrownObject(&site.record))
    {
        return pointers;
    }

    site.pointers.ExceptionRecord = &site.record;
    site.pointers.ContextRecord = &site.context;
    return &site.pointers;
}

// The return addresses from the context of a throw outward, the first the throw's own pc
inline int walk(const CONTEXT *start, uintptr_t *pcs, int max)
{
    auto context = *start;
    auto count = 0;
    __try
    {
#if defined(_M_X64) || defined(_M_ARM64)
        while (count < max)
        {
#ifdef _M_X64
            auto pc = context.Rip;
            auto sp = context.Rsp;
#else
            auto pc = context.Pc;
            auto sp = context.Sp;
#endif
            if (!pc)
            {
                break;
            }

            pcs[count++] = static_cast<uintptr_t>(pc);
            // the JIT's code too: it registers its .pdata (RtlAddFunctionTable)
            DWORD64 imageBase;
            auto *function = RtlLookupFunctionEntry(pc, &imageBase, nullptr);
            if (function)
            {
                void *handlerData;
                DWORD64 establisherFrame;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, pc, function, &context, &handlerData,
                                 &establisherFrame, nullptr);
            }
            else
            {
                // a leaf function: where it returns to is on top of the stack, or in the link register
#ifdef _M_X64
                context.Rip = *reinterpret_cast<DWORD64 *>(context.Rsp);
                context.Rsp += 8;
#else
                context.Pc = context.Lr;
#endif
            }

#ifdef _M_X64
            if (context.Rip == pc && context.Rsp == sp)
#else
            if (context.Pc == pc && context.Sp == sp)
#endif
            {
                break;
            }
        }
#elif defined(_M_IX86)
        auto *dbg = dbgHelp();
        if (!dbg || !dbg->stackWalk || !dbg->functionTableAccess || !dbg->moduleBase)
        {
            return 0;
        }

        STACKFRAME64 frame = {};
        frame.AddrPC.Offset = context.Eip;
        frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Offset = context.Ebp;
        frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Offset = context.Esp;
        frame.AddrStack.Mode = AddrModeFlat;
        while (count < max &&
               dbg->stackWalk(IMAGE_FILE_MACHINE_I386, dbg->process, GetCurrentThread(), &frame, &context, nullptr,
                              dbg->functionTableAccess, dbg->moduleBase, nullptr) &&
               frame.AddrPC.Offset)
        {
            pcs[count++] = static_cast<uintptr_t>(frame.AddrPC.Offset);
        }
#endif
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
    }

    return count;
}

// The module that holds `pc`, null for code no module holds (the JIT's)
inline HMODULE moduleOf(uintptr_t pc)
{
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(pc), &module))
    {
        return nullptr;
    }

    return module;
}

// the system's DLLs, and the C/C++ runtime's wherever it is installed
inline bool isSystemModule(HMODULE module)
{
    char path[MAX_PATH];
    auto length = GetModuleFileNameA(module, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
    {
        return false;
    }

    char systemDirectory[MAX_PATH];
    auto systemLength = GetSystemDirectoryA(systemDirectory, MAX_PATH);
    if (systemLength > 0 && systemLength < MAX_PATH && startsWithNoCase(path, systemDirectory))
    {
        return true;
    }

    auto *name = baseName(path);
    return startsWithNoCase(name, "vcruntime") || startsWithNoCase(name, "ucrtbase") ||
           startsWithNoCase(name, "msvcp") || startsWithNoCase(name, "concrt");
}

// Whether `pc` is in the function that starts at `function` (the same unwind entry; on x86, which has
// none, close behind its start)
inline bool inFunction(uintptr_t pc, const void *function)
{
    if (!function)
    {
        return false;
    }

    auto start = reinterpret_cast<uintptr_t>(function);
#if defined(_M_X64) || defined(_M_ARM64)
    DWORD64 pcBase, startBase;
    auto *pcEntry = RtlLookupFunctionEntry(pc, &pcBase, nullptr);
    auto *startEntry = RtlLookupFunctionEntry(start, &startBase, nullptr);
    return pcEntry && pcEntry == startEntry;
#else
    return pc > start && pc - start < 0x100;
#endif
}

inline bool nameFrame(uintptr_t pc, UncaughtFrameName &name)
{
    auto *dbg = dbgHelp();
    if (!dbg)
    {
        return false;
    }

    char buffer[sizeof(SYMBOL_INFO) + sizeof(name.function)];
    auto *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
    memset(symbol, 0, sizeof(SYMBOL_INFO));
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = sizeof(name.function) - 1;
    DWORD64 displacement = 0;
    if (!dbg->fromAddr(dbg->process, pc, &displacement, symbol))
    {
        return false;
    }

    // an export table's nearest name is as likely another function's: names from debug info only
    IMAGEHLP_MODULE64 module = {};
    module.SizeOfStruct = sizeof(module);
    if (!dbg->moduleInfo(dbg->process, pc, &module) || module.SymType == SymNone ||
        module.SymType == SymExport || module.SymType == SymDeferred)
    {
        return false;
    }

    snprintf(name.function, sizeof(name.function), "%s", symbol->Name);
    IMAGEHLP_LINE64 line = {};
    line.SizeOfStruct = sizeof(line);
    DWORD lineDisplacement = 0;
    if (dbg->lineFromAddr(dbg->process, pc, &lineDisplacement, &line) && line.FileName)
    {
        snprintf(name.file, sizeof(name.file), "%s", line.FileName);
        name.line = line.LineNumber;
    }

    return true;
}

#else // Linux

// The return addresses from here outward
struct WalkState
{
    uintptr_t *pcs;
    int max;
    int count;
};

inline _Unwind_Reason_Code walkFrame(_Unwind_Context *context, void *argument)
{
    auto *state = static_cast<WalkState *>(argument);
    auto pc = static_cast<uintptr_t>(_Unwind_GetIP(context));
    if (!pc || state->count >= state->max)
    {
        return _URC_END_OF_STACK;
    }

    state->pcs[state->count++] = pc;
    return _URC_NO_REASON;
}

inline int walk(uintptr_t *pcs, int max)
{
    WalkState state = {pcs, max, 0};
    _Unwind_Backtrace(walkFrame, &state);
    return state.count;
}

// A module (the executable or a shared library) and its symbol table, read from its file
struct Module
{
    char path[512];
    // where it is loaded: its addresses are `pc - bias`
    uintptr_t bias;
    const char *image;
    size_t size;
    const ElfW(Sym) *symbols;
    size_t symbolCount;
    const char *names;
    bool hasLines;
};

struct FindModule
{
    uintptr_t pc;
    bool found;
    bool isProgram;
    char path[512];
    uintptr_t bias;
};

inline int findModule(dl_phdr_info *info, size_t, void *argument)
{
    auto *find = static_cast<FindModule *>(argument);
    for (auto i = 0; i < info->dlpi_phnum; i++)
    {
        auto &header = info->dlpi_phdr[i];
        auto start = info->dlpi_addr + header.p_vaddr;
        if (header.p_type == PT_LOAD && find->pc >= start && find->pc < start + header.p_memsz)
        {
            find->found = true;
            find->isProgram = !info->dlpi_name || !info->dlpi_name[0];
            snprintf(find->path, sizeof(find->path), "%s", find->isProgram ? "/proc/self/exe" : info->dlpi_name);
            if (find->isProgram)
            {
                // the executable by its own name, as a frame without one says it
                auto length = readlink("/proc/self/exe", find->path, sizeof(find->path) - 1);
                find->path[length > 0 ? length : 0] = '\0';
                if (length <= 0)
                {
                    snprintf(find->path, sizeof(find->path), "/proc/self/exe");
                }
            }
            find->bias = info->dlpi_addr;
            return 1;
        }
    }

    return 0;
}

// Its .symtab (or, stripped, .dynsym), mapped once and kept: the program is ending
inline void readModule(Module &module)
{
    auto file = open(module.path, O_RDONLY | O_CLOEXEC);
    if (file < 0)
    {
        return;
    }

    struct stat status;
    if (fstat(file, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(ElfW(Ehdr))))
    {
        close(file);
        return;
    }

    auto *image = mmap(nullptr, status.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    close(file);
    if (image == MAP_FAILED)
    {
        return;
    }

    module.image = static_cast<const char *>(image);
    module.size = status.st_size;
    auto *header = reinterpret_cast<const ElfW(Ehdr) *>(module.image);
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 || header->e_shoff == 0 ||
        header->e_shoff + header->e_shnum * sizeof(ElfW(Shdr)) > module.size)
    {
        return;
    }

    auto *sections = reinterpret_cast<const ElfW(Shdr) *>(module.image + header->e_shoff);
    auto *sectionNames = header->e_shstrndx < header->e_shnum ? module.image + sections[header->e_shstrndx].sh_offset
                                                              : nullptr;
    const ElfW(Shdr) *symbolSection = nullptr;
    for (auto i = 0; i < header->e_shnum; i++)
    {
        auto &section = sections[i];
        if (section.sh_type == SHT_SYMTAB || (section.sh_type == SHT_DYNSYM && !symbolSection))
        {
            symbolSection = &section;
        }

        if (sectionNames && strcmp(sectionNames + section.sh_name, ".debug_line") == 0)
        {
            module.hasLines = true;
        }
    }

    if (!symbolSection || symbolSection->sh_link >= header->e_shnum ||
        symbolSection->sh_offset + symbolSection->sh_size > module.size)
    {
        return;
    }

    module.symbols = reinterpret_cast<const ElfW(Sym) *>(module.image + symbolSection->sh_offset);
    module.symbolCount = symbolSection->sh_size / sizeof(ElfW(Sym));
    module.names = module.image + sections[symbolSection->sh_link].sh_offset;
}

// The module that holds `pc`, read on first use; null for code no module holds (the JIT's)
inline Module *moduleOf(uintptr_t pc, bool *isProgram = nullptr)
{
    static Module modules[16];
    static int moduleCount;

    FindModule find = {};
    find.pc = pc;
    dl_iterate_phdr(findModule, &find);
    if (!find.found)
    {
        return nullptr;
    }

    if (isProgram)
    {
        *isProgram = find.isProgram;
    }

    for (auto i = 0; i < moduleCount; i++)
    {
        if (modules[i].bias == find.bias && strcmp(modules[i].path, find.path) == 0)
        {
            return &modules[i];
        }
    }

    if (moduleCount == 16)
    {
        return nullptr;
    }

    auto &module = modules[moduleCount++];
    snprintf(module.path, sizeof(module.path), "%s", find.path);
    module.bias = find.bias;
    readModule(module);
    return &module;
}

inline bool isSystemModule(const Module *module)
{
    auto *name = strrchr(module->path, '/');
    name = name ? name + 1 : module->path;
    return startsWith(name, "libc.so") || startsWith(name, "libc-") || startsWith(name, "libstdc++") ||
           startsWith(name, "libgcc_s") || startsWith(name, "libc++") || startsWith(name, "ld-linux") ||
           startsWith(name, "ld-musl") || startsWith(name, "libm.so") || startsWith(name, "libpthread") ||
           startsWith(name, "libunwind");
}

// The function of the symbol table that holds `pc`
inline const char *functionAt(const Module *module, uintptr_t pc)
{
    if (!module->symbols)
    {
        return nullptr;
    }

    auto address = pc - module->bias;
    for (size_t i = 0; i < module->symbolCount; i++)
    {
        auto &symbol = module->symbols[i];
        if (ELF64_ST_TYPE(symbol.st_info) == STT_FUNC && symbol.st_value && address >= symbol.st_value &&
            address < symbol.st_value + (symbol.st_size ? symbol.st_size : 1))
        {
            return module->names + symbol.st_name;
        }
    }

    return nullptr;
}

// The source line from debug info, by `addr2line` - when the module has a .debug_line and the tool is there
inline void lineAt(const Module *module, uintptr_t pc, UncaughtFrameName &name)
{
    if (!module->hasLines || strchr(module->path, '\'') || access("/usr/bin/addr2line", X_OK) != 0)
    {
        return;
    }

    char command[700];
    snprintf(command, sizeof(command), "/usr/bin/addr2line -e '%s' 0x%llx 2>/dev/null", module->path,
             static_cast<unsigned long long>(pc - module->bias));
    auto *pipe = popen(command, "r");
    if (!pipe)
    {
        return;
    }

    char result[600];
    if (fgets(result, sizeof(result), pipe))
    {
        // file:line, or file:line (discriminator N); ??:0 or ??:? without one
        result[strcspn(result, "\r\n")] = '\0';
        auto *discriminator = strstr(result, " (discriminator");
        if (discriminator)
        {
            *discriminator = '\0';
        }

        auto *colon = strrchr(result, ':');
        if (colon && result[0] != '?' && colon[1] >= '1' && colon[1] <= '9')
        {
            *colon = '\0';
            snprintf(name.file, sizeof(name.file), "%s", result);
            name.line = static_cast<unsigned>(strtoul(colon + 1, nullptr, 10));
        }
    }

    pclose(pipe);
}

#endif

// A frame with no name: its module and the offset in it (what a symbolizer takes), or its address
inline void sayUnnamedFrame(FILE *out, uintptr_t pc, uintptr_t lookup)
{
#ifdef _WIN32
    if (auto module = moduleOf(lookup))
    {
        char path[MAX_PATH];
        auto length = GetModuleFileNameA(module, path, MAX_PATH);
        fprintf(out, "    at %s+0x%llx\n", length > 0 && length < MAX_PATH ? baseName(path) : "?",
                static_cast<unsigned long long>(pc - reinterpret_cast<uintptr_t>(module)));
        return;
    }
#else
    if (auto *module = moduleOf(lookup))
    {
        auto *name = strrchr(module->path, '/');
        fprintf(out, "    at %s+0x%llx\n", name ? name + 1 : module->path,
                static_cast<unsigned long long>(pc - module->bias));
        return;
    }
#endif

    fprintf(out, "    at 0x%llx\n", static_cast<unsigned long long>(pc));
}

struct Frame
{
    uintptr_t pc;
    FrameKind kind;
    bool named;
    bool isMain;
    UncaughtFrameName name;
};

} // namespace uncaught_stack

// Where the trace is said from
struct UncaughtStackOptions
{
    // The JIT's own module, whose frames are left out and end the trace: on Windows its HMODULE, on Linux
    // any non-null for the executable
    const void *host = nullptr;
    // Windows: _CxxThrowException as the program calls it - it may be in the program, from a static C runtime
    const void *throwFunction = nullptr;
    // Windows: the program's main, after which the C runtime's startup is left out even with no names
    const void *mainFunction = nullptr;
    // names the JIT's code
    UncaughtFrameNamer namer = nullptr;
};

// Says the frames under "uncaught exception: ...". `pcs` are as walked from the throw (the first one is
// the throw's own pc, the others return addresses).
inline void sayUncaughtStack(FILE *out, const uintptr_t *pcs, int count, const UncaughtStackOptions &options)
{
    using namespace uncaught_stack;

    static Frame frames[uncaughtStackMaxFrames];
    if (count > uncaughtStackMaxFrames)
    {
        count = uncaughtStackMaxFrames;
    }

    for (auto i = 0; i < count; i++)
    {
        auto &frame = frames[i];
        memset(&frame, 0, sizeof(frame));
        frame.pc = pcs[i];
        // a return address is the instruction after the call: the call is the one before it
        auto lookup = i == 0 ? frame.pc : frame.pc - 1;
#ifdef _WIN32
        auto *module = moduleOf(lookup);
        if (module && module == static_cast<HMODULE>(const_cast<void *>(options.host)))
        {
            frame.kind = FrameKind::Host;
        }
        else if (module && isSystemModule(module))
        {
            frame.kind = FrameKind::System;
        }
        else if (inFunction(lookup, options.throwFunction))
        {
            frame.kind = FrameKind::System;
        }
        else if (module)
        {
            frame.isMain = inFunction(lookup, options.mainFunction);
            frame.named = nameFrame(lookup, frame.name);
            // a C runtime of the program's own (--jit-native's DLL): its throw
            if (frame.named && (strcmp(frame.name.function, "_CxxThrowException") == 0 ||
                                strcmp(frame.name.function, "CxxThrowException") == 0))
            {
                frame.kind = FrameKind::System;
            }
        }
        else if (options.namer)
        {
            frame.named = options.namer(lookup, frame.name);
        }
#else
        auto isProgram = false;
        auto *module = moduleOf(lookup, &isProgram);
        if (module && options.host && isProgram)
        {
            frame.kind = FrameKind::Host;
        }
        else if (module && isSystemModule(module))
        {
            frame.kind = FrameKind::System;
        }
        else if (module)
        {
            if (auto *function = functionAt(module, lookup))
            {
                frame.named = true;
                snprintf(frame.name.function, sizeof(frame.name.function), "%s", function);
                lineAt(module, lookup, frame.name);
            }
        }
        else if (options.namer)
        {
            frame.named = options.namer(lookup, frame.name);
        }
#endif
        if (frame.named && strcmp(frame.name.function, "main") == 0)
        {
            frame.isMain = true;
        }
    }

    // The throw, and what said it: on Windows the frames from RaiseException to _CxxThrowException; on Linux
    // this handler, std::terminate() and __cxa_throw (in the C++ runtime, or in tslang's own when it has
    // one linked in), after which the program's frames start.
    auto first = 0;
#ifndef _WIN32
    while (first < count && frames[first].kind != FrameKind::System)
    {
        first++;
    }

    if (first == count)
    {
        first = 0;
    }
#endif
    while (first < count && frames[first].kind != FrameKind::Program)
    {
        first++;
    }

    auto said = 0;
    auto namedAny = false;
    for (auto i = first; i < count; i++)
    {
        auto &frame = frames[i];
        if (frame.kind == FrameKind::Host)
        {
            break;
        }

        if (frame.kind == FrameKind::System)
        {
            continue;
        }

        said++;
        if (frame.named)
        {
            namedAny = true;
            if (frame.name.file[0])
            {
                fprintf(out, "    at %s (%s:%u)\n", frame.name.function, frame.name.file, frame.name.line);
            }
            else
            {
                fprintf(out, "    at %s\n", frame.name.function);
            }
        }
        else
        {
            sayUnnamedFrame(out, frame.pc, i == 0 ? frame.pc : frame.pc - 1);
        }

        // the C runtime's startup, after it, is not the program's
        if (frame.isMain)
        {
            break;
        }
    }

    if (said > 0 && !namedAny)
    {
        fprintf(out, "    (compile with --di for function names and source lines)\n");
    }
}

} // namespace typescript

#endif // TSLANG_DESCRIBES_UNCAUGHT

#endif // TYPESCRIPT_UNCAUGHTSTACKTRACE_H_
