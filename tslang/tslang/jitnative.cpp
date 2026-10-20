// --jit-native: the objects of the JIT cache linked into one DLL with its PDB and loaded by the
// system loader instead of RTDyld, so that WinDbg and Visual Studio see the program as a module
// with symbols and source lines. One DLL for all of them, as `--emit=exe` links one program: an
// import of another module's data (a global, a vtable, a class's `.size`) is resolved at link time,
// where a DLL per module would need `__imp_` references nothing emits. See
// docs/jit-native-dll-debugging-design.md.

#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Support/raw_ostream.h"

#include "jitcache.h"
#include "link.h"

#include <chrono>

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

extern cl::opt<bool> enableOpt;

std::string getDefaultLibBinary(bool, const CompileOptions &, const llvm::Triple &);

namespace
{

std::string hashString(llvm::StringRef data)
{
    return llvm::utohexstr(llvm::xxh3_64bits(data), /*LowerCase=*/true, /*Width=*/16);
}

// The functions the DLL exports for tslang beside the program's own, in an object of their own: the
// cached objects are also the ones the in-memory JIT loads, where none of this applies.
//
// The DLL links the C runtime statically, so it has its own: its own stdout buffer, which nothing in
// tslang can flush, and its own error modes, which no executable's startup has set - a DLL's C
// runtime does not know it runs in a console program, and reports a failing assert or an abort()
// with a modal box, a hang in an unattended run.
//   - JIT_NATIVE_FLUSH_NAME flushes every stream of it; tslang calls it after the program, before
//     TerminateProcess, which flushes nothing.
//   - JIT_NATIVE_INIT_NAME sends assert and abort reports to stderr and takes the dialogs away
//     (the debug C runtime's report modes too, if that is the one linked), and has SIGABRT flush
//     the program's output first: a failing assert ends in abort(), which flushes nothing either.
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> compileHostFunctions(llvm::TargetMachine &tm, bool debugCRT)
{
    llvm::LLVMContext context;
    llvm::Module module("tslang_jit_native", context);
    auto dataLayout = tm.createDataLayout();
    module.setDataLayout(dataLayout);
    module.setTargetTriple(tm.getTargetTriple());

    auto *voidType = llvm::Type::getVoidTy(context);
    auto *i32Type = llvm::Type::getInt32Ty(context);
    auto *ptrType = llvm::PointerType::getUnqual(context);
    auto *nullPtr = llvm::ConstantPointerNull::get(ptrType);
    auto fflushFn = module.getOrInsertFunction("fflush", i32Type, ptrType);

    auto define = [&](llvm::StringRef name, llvm::FunctionType *type, llvm::GlobalValue::LinkageTypes linkage) {
        auto *function = llvm::Function::Create(type, linkage, name, module);
        function->setUWTableKind(llvm::UWTableKind::Async);
        return function;
    };

    auto *flush = define(JIT_NATIVE_FLUSH_NAME, llvm::FunctionType::get(voidType, false), llvm::GlobalValue::ExternalLinkage);
    {
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", flush));
        builder.CreateCall(fflushFn, {nullPtr});
        builder.CreateRetVoid();
    }

    auto *onAbort = define("__tslang_jit_native_on_abort", llvm::FunctionType::get(voidType, {i32Type}, false),
                           llvm::GlobalValue::InternalLinkage);
    {
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", onAbort));
        builder.CreateCall(fflushFn, {nullPtr});
        builder.CreateRetVoid();
    }

    auto *init = define(JIT_NATIVE_INIT_NAME, llvm::FunctionType::get(voidType, false), llvm::GlobalValue::ExternalLinkage);
    {
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", init));
        // _set_error_mode(_OUT_TO_STDERR)
        builder.CreateCall(module.getOrInsertFunction("_set_error_mode", i32Type, i32Type), {builder.getInt32(1)});
        // _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT)
        builder.CreateCall(module.getOrInsertFunction("_set_abort_behavior", i32Type, i32Type, i32Type),
                           {builder.getInt32(0), builder.getInt32(3)});
        // signal(SIGABRT, onAbort)
        builder.CreateCall(module.getOrInsertFunction("signal", ptrType, i32Type, ptrType), {builder.getInt32(22), onAbort});
        if (debugCRT)
        {
            auto setReportMode = module.getOrInsertFunction("_CrtSetReportMode", i32Type, i32Type, i32Type);
            auto setReportFile = module.getOrInsertFunction("_CrtSetReportFile", ptrType, i32Type, ptrType);
            // _CRTDBG_FILE_STDERR
            auto *stderrFile = llvm::ConstantExpr::getIntToPtr(
                llvm::ConstantInt::get(dataLayout.getIntPtrType(context), -5, /*IsSigned=*/true), ptrType);
            // _CRT_ERROR (abort's report), _CRT_ASSERT
            for (auto reportType : {1, 2})
            {
                // _CRTDBG_MODE_FILE
                builder.CreateCall(setReportMode, {builder.getInt32(reportType), builder.getInt32(1)});
                builder.CreateCall(setReportFile, {builder.getInt32(reportType), stderrFile});
            }
        }

        builder.CreateRetVoid();
    }

    llvm::orc::SimpleCompiler compiler(tm);
    return compiler(module);
}

// The DLLs of earlier runs of the program, linked from objects or options since changed: removed,
// but for any a run still has loaded, which cannot be. Not another run's DLL still being linked -
// unless it is an hour old: what a run killed while it linked (by a debugger, say) leaves behind.
void removeOtherDlls(llvm::StringRef dllPath, llvm::StringRef prefix)
{
    auto staleBefore = std::chrono::system_clock::now() - std::chrono::hours(1);
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(llvm::sys::path::parent_path(dllPath), ec), end; it != end && !ec;
         it.increment(ec))
    {
        auto path = it->path();
        auto name = llvm::sys::path::filename(path);
        auto extension = llvm::sys::path::extension(path);
        if (path == dllPath || !name.starts_with(prefix))
        {
            continue;
        }

        if (name.contains(".tmp.") || name.ends_with(".host.obj"))
        {
            llvm::sys::fs::file_status status;
            if (!llvm::sys::fs::status(path, status) && status.getLastModificationTime() < staleBefore)
            {
                llvm::sys::fs::remove(path);
            }

            continue;
        }

        if (extension != ".dll" && extension != ".pdb")
        {
            continue;
        }

        llvm::SmallString<256> dllOfPath(path);
        llvm::sys::path::replace_extension(dllOfPath, "dll");
        if (dllOfPath != dllPath)
        {
            llvm::sys::fs::remove(path);
        }
    }
}

} // namespace

int linkJitUnits(const char *argv0, llvm::ArrayRef<JitUnit> units, CompileOptions &compileOptions,
                 llvm::TargetMachine &tm, std::string &dllPath)
{
    for (auto &unit : units)
    {
        if (unit.objectPath.empty())
        {
            llvm::WithColor::error(llvm::errs(), "tslang")
                << "--jit-native links the objects in the JIT cache, and '" << unit.sourcePath
                << "' has none: its cache folder could not be written (see --jit-cache-dir)\n";
            return -1;
        }
    }

    auto &program = units.back();

    llvm::SmallVector<std::string> exports{JIT_ENTRY_THUNK_NAME, JIT_NATIVE_INIT_NAME, JIT_NATIVE_FLUSH_NAME};
    for (auto &unit : units)
    {
        if (!unit.initFunction.empty())
        {
            exports.push_back(unit.initFunction);
        }

        if (unit.isMain && unit.hasGCtorsMethod)
        {
            exports.push_back(MLIR_GCTORS);
        }
    }

    // the C runtime linkBinary links
    auto debugCRT = !enableOpt.getValue();

    // Everything the DLL is linked from: the objects (by their hashes), what it exports, and the
    // options linkBinary picks the libraries by. A rebuilt compiler may link other libraries.
    std::string key;
    {
        llvm::raw_string_ostream os(key);
        os << "tslang-jit-native 1\n";
        os << "compiler " << jitCompilerId(argv0) << "\n";
        for (auto &unit : units)
        {
            os << "object " << unit.objectHash << "\n";
        }

        for (auto &name : exports)
        {
            os << "export " << name << "\n";
        }

        os << "crt=" << (debugCRT ? "debug" : "release") << " di=" << compileOptions.generateDebugInfo
           << " mm=" << compileOptions.memoryModel << " no-default-lib=" << compileOptions.noDefaultLib << "\n";
        os << "default-lib="
           << (compileOptions.noDefaultLib ? std::string() : getDefaultLibBinary(/*shared=*/false, compileOptions, tm.getTargetTriple()))
           << "\n";
    }

    auto prefix = llvm::sys::path::filename(program.sourcePath).str() + ".";
    llvm::SmallString<256> path(llvm::sys::path::parent_path(program.objectPath));
    llvm::sys::path::append(path, prefix + hashString(key) + ".dll");
    dllPath = path.str().str();

    llvm::SmallString<256> pdbPath(dllPath);
    llvm::sys::path::replace_extension(pdbPath, "pdb");

    if (llvm::sys::fs::exists(dllPath) && (!compileOptions.generateDebugInfo || llvm::sys::fs::exists(pdbPath)))
    {
        LLVM_DEBUG(llvm::dbgs() << "JIT native: using " << dllPath << "\n";);
        return 0;
    }

    auto hostFunctions = compileHostFunctions(tm, debugCRT);
    if (!hostFunctions)
    {
        auto err = hostFunctions.takeError();
        llvm::WithColor::error(llvm::errs(), "tslang") << "--jit-native: failed to compile the host functions, error: " << err << "\n";
        llvm::consumeError(std::move(err));
        return -1;
    }

    // Written and linked under names of their own, then the DLL renamed into place: another run
    // linking the same program at the same time never loads half a file.
    llvm::SmallString<256> hostObjectPath;
    llvm::SmallString<256> tempDllPath;
    if (llvm::sys::fs::createUniqueFile(dllPath + ".%%%%%%%%.host.obj", hostObjectPath) ||
        llvm::sys::fs::createUniqueFile(dllPath + ".%%%%%%%%.tmp.dll", tempDllPath))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "--jit-native: could not create a file beside " << dllPath << "\n";
        return -1;
    }

    llvm::FileRemover hostObjectRemover(hostObjectPath);
    llvm::FileRemover tempDllRemover(tempDllPath);
    // the import library and its exports file the link writes beside the DLL, which nothing uses
    llvm::SmallString<256> tempLibPath(tempDllPath);
    llvm::sys::path::replace_extension(tempLibPath, "lib");
    llvm::FileRemover tempLibRemover(tempLibPath);
    llvm::SmallString<256> tempExpPath(tempDllPath);
    llvm::sys::path::replace_extension(tempExpPath, "exp");
    llvm::FileRemover tempExpRemover(tempExpPath);

    {
        std::error_code ec;
        llvm::raw_fd_ostream os(hostObjectPath, ec);
        if (!ec)
        {
            os << (*hostFunctions)->getBuffer();
            os.close();
        }

        if (ec || os.has_error())
        {
            os.clear_error();
            llvm::WithColor::error(llvm::errs(), "tslang") << "--jit-native: could not write " << hostObjectPath << "\n";
            return -1;
        }
    }

    LinkRequest request;
    for (auto &unit : units)
    {
        request.objects.push_back(unit.objectPath);
    }

    request.objects.push_back(hostObjectPath.str().str());
    request.output = tempDllPath.str().str();
    request.shared = true;
    // The static default library, linked in as a program links it. Its DLL would do for functions,
    // but not for its data: a static field the program reads (a generic's instantiation reading
    // PrimeHelpers.hashCollisionThreshold, say) is referenced by its own name, which only an
    // `__imp_` reference resolves to an import. The in-memory JIT binds it from the export table.
    request.staticDefaultLib = true;
    request.quiet = true;
    request.extraArgs.push_back("-Wl,/INCREMENTAL:NO");
    for (auto &name : exports)
    {
        request.extraArgs.push_back("-Wl,/EXPORT:" + name);
    }

    if (compileOptions.generateDebugInfo)
    {
        // the PDB by its final name, and recorded in the DLL by its name alone, so the debugger
        // finds it beside the DLL however the DLL was renamed or moved
        request.extraArgs.push_back("-Wl,/PDB:" + pdbPath.str().str());
        request.extraArgs.push_back("-Wl,/PDBALTPATH:%_PDB%");
    }

    LLVM_DEBUG(llvm::dbgs() << "JIT native: linking " << dllPath << "\n";);
    if (auto result = linkBinary(argv0, request, compileOptions))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "--jit-native: failed to link " << dllPath << "\n";
        return result;
    }

    if (auto error = llvm::sys::fs::rename(tempDllPath, dllPath))
    {
        // another run linked it first, and has it loaded
        if (!llvm::sys::fs::exists(dllPath))
        {
            llvm::WithColor::error(llvm::errs(), "tslang")
                << "--jit-native: could not rename " << tempDllPath << " to " << dllPath << ": " << error.message() << "\n";
            return -1;
        }
    }
    else
    {
        tempDllRemover.releaseFile();
    }

    removeOtherDlls(dllPath, prefix);
    return 0;
}
