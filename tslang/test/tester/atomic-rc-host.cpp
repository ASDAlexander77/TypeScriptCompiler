// A host that loads a `-mm=rc` shared library and calls into it from several threads at once, all
// of them taking and dropping references to the same object. See atomic-rc.cmake.
//
//   atomic-rc-host <library> <threads> <calls>
//
// Each thread calls churn(calls), which takes and drops <calls> references to the library's one
// shared object. Afterwards the object's count has to be what it was before. Exits 0 when it is,
// 2 when it is not.

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using ChurnFn = double (*)(double);
using CountFn = double (*)();

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <library> <threads> <calls>\n", argv[0]);
        return 1;
    }

    auto threads = std::atoi(argv[2]);
    auto calls = static_cast<double>(std::atoi(argv[3]));

#ifdef _WIN32
    auto library = LoadLibraryExA(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!library)
    {
        std::printf("LoadLibrary '%s' failed: %lu\n", argv[1], GetLastError());
        return 1;
    }

    auto churn = reinterpret_cast<ChurnFn>(GetProcAddress(library, "churn"));
    auto count = reinterpret_cast<CountFn>(GetProcAddress(library, "count"));
#else
    auto library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
    {
        std::printf("dlopen: %s\n", dlerror());
        return 1;
    }

    auto churn = reinterpret_cast<ChurnFn>(dlsym(library, "churn"));
    auto count = reinterpret_cast<CountFn>(dlsym(library, "count"));
#endif
    if (!churn || !count)
    {
        std::printf("'%s' has no 'churn' or no 'count'\n", argv[1]);
        return 1;
    }

    auto before = count();

    std::vector<std::thread> pool;
    for (int id = 0; id < threads; id++)
    {
        pool.emplace_back([&]() {
            if (churn(calls) != calls)
            {
                std::printf("churn read the shared object wrong\n");
                std::fflush(stdout);
                std::exit(3);
            }
        });
    }

    for (auto &thread : pool)
    {
        thread.join();
    }

    auto after = count();
    std::printf("%d threads, %g calls each: count %g before, %g after\n", threads, calls, before, after);
    return after == before ? 0 : 2;
}
