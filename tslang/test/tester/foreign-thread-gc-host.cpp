// A host that is not a tslang program - an Android app, a C program - loading a tslang shared
// library and calling it from threads of its own, which the collector has never heard of. See
// foreign-thread-gc.cmake.
//
//   foreign-thread-gc-host <library> <threads> <calls>
//
// Calls work(seed) <calls> times on the loading thread, then on each of <threads> new threads,
// and checks every result. Exits 0 when all are right.

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using WorkFn = double (*)(double);

static WorkFn work;
static int calls;

// what work(seed) in foreign-thread-gc.cmake returns: the sum of seed + i for i in [0, 200)
static double expected(double seed)
{
    return 200 * seed + 199 * 200 / 2;
}

static void run(long id)
{
    for (int i = 0; i < calls; i++)
    {
        auto seed = static_cast<double>(id * 100000 + i);
        auto got = work(seed);
        if (got != expected(seed))
        {
            std::printf("thread %ld, call %d: got %g, expected %g\n", id, i, got, expected(seed));
            std::fflush(stdout);
            std::exit(2);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <library> <threads> <calls>\n", argv[0]);
        return 1;
    }

    auto threads = std::atoi(argv[2]);
    calls = std::atoi(argv[3]);

#ifdef _WIN32
    // the library's own directory first, where tslang put gc.dll beside it
    auto library = LoadLibraryExA(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!library)
    {
        std::printf("LoadLibrary '%s' failed: %lu\n", argv[1], GetLastError());
        return 1;
    }

    work = reinterpret_cast<WorkFn>(GetProcAddress(library, "work"));
#else
    auto library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
    {
        std::printf("dlopen: %s\n", dlerror());
        return 1;
    }

    work = reinterpret_cast<WorkFn>(dlsym(library, "work"));
#endif
    if (!work)
    {
        std::printf("'%s' has no 'work'\n", argv[1]);
        return 1;
    }

    run(0);
    std::printf("loading thread: %d calls\n", calls);

    std::vector<std::thread> pool;
    for (long id = 1; id <= threads; id++)
    {
        pool.emplace_back(run, id);
    }

    for (auto &thread : pool)
    {
        thread.join();
    }

    std::printf("%d other threads: %d calls each\n", threads, calls);
    return 0;
}
