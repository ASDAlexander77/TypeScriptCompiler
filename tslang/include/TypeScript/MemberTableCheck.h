// The check a program makes of a shared library's classes when it loads the library (ts.CheckMembers,
// mlirGenImportSharedLib): the program reads each class's fields at the offsets it was compiled with, and
// a library built again with another layout would be read wrongly - silently. Each class's member table
// (MEMBERS_NAME in Defines.h) says where the library has them; the program's own view of it
// (MEMBERS_EXPECTED_NAME) says where it reads them. Any field moved, gone or of another type, or another
// instance size, stops the program with a message naming it.
//
// Header-only: an executable gets it from TypeScriptAsyncRuntime (DynamicRuntime.cpp), the JIT from
// jit.cpp, which defines the shared-library entry points itself.

#ifndef TYPESCRIPT_MEMBER_TABLE_CHECK_H
#define TYPESCRIPT_MEMBER_TABLE_CHECK_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace typescript_member_table
{

// MEMBERS_NAME's layout: {count: i32, flags: i32, keys: ptr, entries: ptr}, each entry {name, kind, access,
// readonly, offset, size, hash} with offset and size pointer-sized
struct Entry
{
    const char *name;
    int32_t kind;
    int32_t access;
    int32_t readOnly;
    intptr_t offset;
    intptr_t size;
    int64_t hash;
};

struct Header
{
    int32_t count;
    int32_t flags;
    void *keys;
    const Entry *entries;
};

// MEMBER_KIND_FIELD
constexpr int32_t KindField = 0;

inline const Entry *find(const Header *table, const char *name)
{
    for (int32_t i = 0; i < table->count; i++)
    {
        if (std::strcmp(table->entries[i].name, name) == 0)
        {
            return &table->entries[i];
        }
    }

    return nullptr;
}

[[noreturn]] inline void stop(const char *library, const char *className, const char *member, const char *what)
{
    if (member)
    {
        std::fprintf(stderr, "library '%s': %s.%s %s; rebuild this program against the new library\n", library, className,
                     member, what);
    }
    else
    {
        std::fprintf(stderr, "library '%s': %s %s; rebuild this program against the new library\n", library, className, what);
    }

    std::fflush(stderr);
    std::exit(1);
}

inline void check(const void *expectedTable, const void *actualTable, const char *library, const char *className,
                  intptr_t expectedSize, const void *actualSizeAddress)
{
    auto expected = static_cast<const Header *>(expectedTable);
    auto actual = static_cast<const Header *>(actualTable);
    // a class the library does not export (one its exports name), a @value or a @cpp class: no table to find.
    // A library from before tables is told by its declarations (MEMBERS_MARKER) and not checked at all.
    if (!actual)
    {
        return;
    }

    char what[128];
    for (int32_t i = 0; expected && i < expected->count; i++)
    {
        auto &field = expected->entries[i];
        if (field.kind != KindField)
        {
            continue;
        }

        auto found = find(actual, field.name);
        if (!found || found->kind != KindField)
        {
            stop(library, className, field.name, "is missing");
        }

        if (found->offset != field.offset)
        {
            std::snprintf(what, sizeof(what), "moved from offset %lld to %lld", (long long)field.offset, (long long)found->offset);
            stop(library, className, field.name, what);
        }

        if (found->hash != field.hash || found->size != field.size)
        {
            stop(library, className, field.name, "changed type");
        }
    }

    if (actualSizeAddress)
    {
        auto actualSize = *static_cast<const int64_t *>(actualSizeAddress);
        if (sizeof(void *) == 4)
        {
            actualSize = *static_cast<const int32_t *>(actualSizeAddress);
        }

        if (actualSize != expectedSize)
        {
            std::snprintf(what, sizeof(what), "changed size from %lld to %lld bytes", (long long)expectedSize, (long long)actualSize);
            stop(library, className, nullptr, what);
        }
    }
}

} // namespace typescript_member_table

#endif // TYPESCRIPT_MEMBER_TABLE_CHECK_H
