// Stack sampler: runs a command, samples its main thread's call stack, prints exclusive and
// inclusive function counts. Usage: sampler.exe <interval_ms> <command line...>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

static std::string symName(HANDLE proc, DWORD64 addr)
{
    char buf[sizeof(SYMBOL_INFO) + 1024];
    auto *sym = (SYMBOL_INFO *)buf;
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 1000;
    DWORD64 disp = 0;
    if (SymFromAddr(proc, addr, &disp, sym))
    {
        std::string s(sym->Name);
        if (s.size() > 170) s = s.substr(0, 170) + "...";
        return s;
    }
    char tmp[64];
    sprintf(tmp, "0x%llx", (unsigned long long)addr);
    return tmp;
}

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    int interval = atoi(argv[1]);
    std::string cmd;
    for (int i = 2; i < argc; i++)
    {
        if (i > 2) cmd += " ";
        cmd += std::string("\"") + argv[i] + "\"";
    }

    STARTUPINFOA si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    HANDLE nul = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    SetHandleInformation(nul, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = nul;
    si.hStdError = nul;
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi))
    {
        printf("CreateProcess failed %lu\n", GetLastError());
        return 1;
    }
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    ResumeThread(pi.hThread);
    Sleep(150);
    SymInitialize(pi.hProcess, nullptr, TRUE);

    std::vector<std::vector<DWORD64>> stacks;
    while (WaitForSingleObject(pi.hProcess, interval) == WAIT_TIMEOUT)
    {
        if (SuspendThread(pi.hThread) == (DWORD)-1) break;
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_FULL;
        std::vector<DWORD64> st;
        if (GetThreadContext(pi.hThread, &ctx))
        {
            STACKFRAME64 f{};
            f.AddrPC.Offset = ctx.Rip;  f.AddrPC.Mode = AddrModeFlat;
            f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
            f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
            for (int i = 0; i < 96; i++)
            {
                if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, pi.hProcess, pi.hThread, &f, &ctx, nullptr,
                                 SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
                    break;
                if (!f.AddrPC.Offset) break;
                st.push_back(f.AddrPC.Offset);
            }
        }
        ResumeThread(pi.hThread);
        if (!st.empty()) stacks.push_back(std::move(st));
    }

    std::map<DWORD64, std::string> names;
    auto name = [&](DWORD64 a) -> const std::string & {
        auto it = names.find(a);
        if (it == names.end()) it = names.emplace(a, symName(pi.hProcess, a)).first;
        return it->second;
    };
    std::map<std::string, int> excl, incl;
    std::map<std::string, std::map<std::string, int>> leafStacks;
    for (auto &st : stacks)
    {
        excl[name(st[0])]++;
        std::set<std::string> seen;
        for (auto a : st) seen.insert(name(a));
        for (auto &s : seen) incl[s]++;
        std::string key;
        for (size_t i = 1; i < st.size() && i < 14; i++) key += "      <- " + name(st[i]) + "\n";
        leafStacks[name(st[0])][key]++;
    }
    auto top = [&](std::map<std::string, int> &m, int n, const char *title) {
        std::vector<std::pair<int, std::string>> v;
        for (auto &kv : m) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        printf("\n== %s (%zu samples) ==\n", title, stacks.size());
        for (int i = 0; i < n && i < (int)v.size(); i++)
            printf("%6.1f%%  %s\n", 100.0 * v[i].first / stacks.size(), v[i].second.c_str());
        return v;
    };
    auto ex = top(excl, 30, "exclusive");
    top(incl, 70, "inclusive");
    printf("\n== most common stack of the top 6 leaves ==\n");
    for (int i = 0; i < 6 && i < (int)ex.size(); i++)
    {
        auto &m = leafStacks[ex[i].second];
        auto best = std::max_element(m.begin(), m.end(), [](auto &a, auto &b) { return a.second < b.second; });
        printf("%s  (%d of %d samples with this stack)\n%s\n", ex[i].second.c_str(), best->second, ex[i].first, best->first.c_str());
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    printf("exit code %lu\n", code);
    SymCleanup(pi.hProcess);
    return 0;
}
