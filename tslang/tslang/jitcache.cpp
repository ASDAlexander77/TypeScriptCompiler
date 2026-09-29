// The JIT cache: every .ts file a JIT run needs - the program and each .ts module it imports - is
// compiled into an object of its own and kept in a cache folder, `__jit` beside the source (or
// the folder given with --jit-cache-dir). The next run loads the object instead of compiling the
// file again, as long as nothing it was compiled from has changed.
//
// Next to each object is a manifest (<object>.deps) saying what the object was compiled from:
//
//   tslang-jit-cache 1
//   compiler <the tslang executable's size and time>
//   object <hash of the object>
//   init <the function the file's global constructors were moved into>
//   gctors <1 if the file has __mlir_gctors>
//   import <a .ts module the file imports, whose object has to be loaded beside it>
//   dep <hash> <a file it was compiled from: itself, what it references and imports, lib.d.ts>
//
// An object is used only if every one of these still holds. The name of an object is the file's
// name and a hash of its path and the options it was compiled with, so a run with other options
// has objects of its own and does not replace these.

#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "jitcache.h"

#include <algorithm>

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::opt<std::string> mainFuncName;
extern cl::opt<bool> verifyOwnership;
extern cl::opt<bool> ownSkipInference;
extern cl::opt<std::string> jitCacheDir;

std::unique_ptr<mlir::MLIRContext> createMLIRContext();
int compileTypeScriptFileIntoMLIR(mlir::MLIRContext &, llvm::StringRef, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &, CompileOptions &);
int runMLIRPasses(mlir::MLIRContext &, llvm::SourceMgr &, mlir::OwningOpRef<mlir::ModuleOp> &, CompileOptions &);
int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int, CompileOptions &);
llvm::Error addEntryThunk(llvm::Module &, llvm::StringRef);

#define JIT_CACHE_FORMAT "tslang-jit-cache 1"
#define JIT_CACHE_DIR "__jit"
#define JIT_INIT_PREFIX "__tslang_jit_init_"

namespace
{

std::string hashString(llvm::StringRef data)
{
    return llvm::utohexstr(llvm::xxh3_64bits(data), /*LowerCase=*/true, /*Width=*/16);
}

std::string canonicalPath(llvm::StringRef path)
{
    llvm::SmallString<256> canonical;
    if (!llvm::sys::fs::real_path(path, canonical))
    {
        return canonical.str().str();
    }

    canonical = path;
    llvm::sys::fs::make_absolute(canonical);
    llvm::sys::path::remove_dots(canonical, /*remove_dot_dot=*/true);
    return canonical.str().str();
}

// Written beside the final name and renamed over it, so a run reading the cache - another test
// running the same module at the same time, say - never sees half a file.
bool writeFileAtomically(llvm::StringRef path, llvm::StringRef content)
{
    int fd;
    llvm::SmallString<256> tempPath;
    if (llvm::sys::fs::createUniqueFile(path + ".%%%%%%%%.tmp", fd, tempPath))
    {
        return false;
    }

    {
        llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
        os << content;
        os.close();
        if (os.has_error())
        {
            os.clear_error();
            llvm::sys::fs::remove(tempPath);
            return false;
        }
    }

    if (llvm::sys::fs::rename(tempPath, path))
    {
        llvm::sys::fs::remove(tempPath);
        return false;
    }

    return true;
}

// The JIT runs no object's llvm.global_ctors - only an IR module's, whose constructors its
// platform takes out before compiling it. So they are taken out here, into one function, in the
// order of their priority; the JIT calls it before the program starts (runJitProgram).
std::string moveGlobalCtorsIntoFunction(llvm::Module &llvmModule, llvm::StringRef functionName)
{
    auto *globalCtors = llvmModule.getNamedGlobal("llvm.global_ctors");
    if (!globalCtors)
    {
        return "";
    }

    struct Ctor
    {
        unsigned priority;
        llvm::Function *function;
    };

    std::vector<Ctor> ctors;
    for (auto ctor : llvm::orc::getConstructors(llvmModule))
    {
        if (ctor.Func)
        {
            ctors.push_back({ctor.Priority, ctor.Func});
        }
    }

    globalCtors->eraseFromParent();
    if (ctors.empty())
    {
        return "";
    }

    std::stable_sort(ctors.begin(), ctors.end(), [](const Ctor &a, const Ctor &b) { return a.priority < b.priority; });

    auto &context = llvmModule.getContext();
    auto *initFunction = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
                                                llvm::Function::ExternalLinkage, functionName, llvmModule);
    // a constructor may throw through it; Win64 can only unwind a frame it has unwind info for
    initFunction->setUWTableKind(llvm::UWTableKind::Async);

    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", initFunction));
    for (auto &ctor : ctors)
    {
        auto *call = builder.CreateCall(ctor.function->getFunctionType(), ctor.function);
        call->setCallingConv(ctor.function->getCallingConv());
    }

    builder.CreateRetVoid();
    return functionName.str();
}

// A definition several objects have - a class's `.size`, say, in the module declaring the class
// and in every module importing it - is put in a COMDAT, and a linker keeps one of them. RTDyld
// does that for COFF (ORC takes a COMDAT symbol for a weak one) but not for ELF, where the second
// object loaded fails with "duplicate definition". Made weak, ORC keeps the first.
void weakenComdatDefinitions(llvm::Module &llvmModule)
{
    if (!llvmModule.getTargetTriple().isOSBinFormatELF())
    {
        return;
    }

    for (auto &globalObject : llvmModule.global_objects())
    {
        if (globalObject.hasComdat() && !globalObject.isDeclaration() && globalObject.hasExternalLinkage())
        {
            globalObject.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
        }
    }
}

struct Manifest
{
    std::string compiler;
    std::string objectHash;
    std::string initFunction;
    bool hasGCtorsMethod = false;
    std::vector<std::string> imports;
    // hash, path
    std::vector<std::pair<std::string, std::string>> deps;

    std::string str() const
    {
        std::string text;
        llvm::raw_string_ostream os(text);
        os << JIT_CACHE_FORMAT "\n";
        os << "compiler " << compiler << "\n";
        os << "object " << objectHash << "\n";
        os << "init " << (initFunction.empty() ? "-" : initFunction) << "\n";
        os << "gctors " << (hasGCtorsMethod ? 1 : 0) << "\n";
        for (auto &import : imports)
        {
            os << "import " << import << "\n";
        }

        for (auto &[hash, path] : deps)
        {
            os << "dep " << hash << " " << path << "\n";
        }

        return text;
    }

    bool parse(llvm::StringRef text)
    {
        llvm::SmallVector<llvm::StringRef> lines;
        text.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        if (lines.empty() || lines.front().rtrim() != JIT_CACHE_FORMAT)
        {
            return false;
        }

        for (auto line : llvm::ArrayRef(lines).drop_front())
        {
            line = line.rtrim("\r");
            auto [key, value] = line.split(' ');
            if (key == "compiler")
            {
                compiler = value.str();
            }
            else if (key == "object")
            {
                objectHash = value.str();
            }
            else if (key == "init")
            {
                initFunction = value == "-" ? "" : value.str();
            }
            else if (key == "gctors")
            {
                hasGCtorsMethod = value == "1";
            }
            else if (key == "import")
            {
                imports.push_back(value.str());
            }
            else if (key == "dep")
            {
                auto [hash, path] = value.split(' ');
                deps.push_back({hash.str(), path.str()});
            }
            else
            {
                return false;
            }
        }

        return !compiler.empty() && !objectHash.empty();
    }
};

class JitUnitBuilder
{
  public:
    JitUnitBuilder(const char *argv0, const CompileOptions &compileOptions, llvm::orc::JITTargetMachineBuilder &tmBuilder,
                   std::vector<JitUnit> &units)
        : compileOptions(compileOptions), tmBuilder(tmBuilder), units(units)
    {
        // a rebuilt compiler compiles differently: its objects are not these
        auto executable = llvm::sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&canonicalPath));
        llvm::sys::fs::file_status status;
        if (!executable.empty() && !llvm::sys::fs::status(executable, status))
        {
            compilerId = llvm::utohexstr(status.getSize(), true) + "-" +
                         llvm::utohexstr(status.getLastModificationTime().time_since_epoch().count(), true);
        }
    }

    int add(llvm::StringRef sourcePath, bool isMain)
    {
        auto path = canonicalPath(sourcePath);
        if (!visited.insert(path).second)
        {
            // loaded already, or on the way: an import cycle, or a module importing the program
            return 0;
        }

        if (!tm)
        {
            auto tmOrError = tmBuilder.createTargetMachine();
            if (!tmOrError)
            {
                auto err = tmOrError.takeError();
                llvm::WithColor::error(llvm::errs(), "tslang") << "failed to create a TargetMachine for the host, error: " << err << "\n";
                llvm::consumeError(std::move(err));
                return -1;
            }

            tm = std::move(*tmOrError);
        }

        JitUnit unit;
        unit.sourcePath = path;
        unit.isMain = isMain;

        auto [objectPath, manifestPath] = cachePaths(path, isMain);

        Manifest manifest;
        if (!loadFromCache(objectPath, manifestPath, manifest, unit))
        {
            if (auto result = compile(path, isMain, manifest, unit))
            {
                return result;
            }

            storeInCache(objectPath, manifestPath, manifest, *unit.object);
        }

        unit.initFunction = manifest.initFunction;
        unit.hasGCtorsMethod = manifest.hasGCtorsMethod;

        // what it imports first: their objects are loaded, and their constructors run, before its own
        for (auto &import : manifest.imports)
        {
            if (auto result = add(import, /*isMain=*/false))
            {
                return result;
            }
        }

        units.push_back(std::move(unit));
        return 0;
    }

  private:
    // Everything that changes what a file compiles into, but the files themselves (the manifest's
    // deps) and the compiler (its `compiler`).
    std::string optionsKey(bool isMain)
    {
        std::string key;
        llvm::raw_string_ostream os(key);
        os << (isMain ? "main " + mainFuncName.getValue() : std::string("import")) << "\n";
        os << "mm=" << compileOptions.memoryModel << " opt=" << enableOpt.getValue() << " opt_level=" << optLevel.getValue()
           << " size_level=" << sizeLevel.getValue() << " di=" << compileOptions.generateDebugInfo
           << " lldb=" << compileOptions.lldbDebugInfo << " no-default-lib=" << compileOptions.noDefaultLib
           << " builtins=" << compileOptions.enableBuiltins << " strict-null-checks=" << compileOptions.strictNullChecks
           << " fast-math=" << compileOptions.enableFastMath << " gctors-as-method=" << compileOptions.appendGCtorsToMethod
           << " embed-declarations=" << compileOptions.embedExportDeclarations
           << " verify-ownership=" << verifyOwnership.getValue() << " own-skip-inference=" << ownSkipInference.getValue()
           << "\n";
        os << "export=" << llvm::join(compileOptions.exportFilters, ",") << "\n";
        os << "default-lib=" << compileOptions.defaultDeclarationTSFile << "\n";
        os << "target=" << tmBuilder.getTargetTriple().str() << " cpu=" << tmBuilder.getCPU()
           << " features=" << tmBuilder.getFeatures().getString() << "\n";
        return key;
    }

    std::pair<std::string, std::string> cachePaths(llvm::StringRef path, bool isMain)
    {
        llvm::SmallString<256> objectPath;
        if (jitCacheDir.empty())
        {
            objectPath = llvm::sys::path::parent_path(path);
            llvm::sys::path::append(objectPath, JIT_CACHE_DIR);
        }
        else
        {
            objectPath = jitCacheDir.getValue();
        }

        auto name = llvm::sys::path::filename(path).str() + "." + hashString(path.str() + "\n" + optionsKey(isMain));
        name += tm->getTargetTriple().isOSBinFormatCOFF() ? ".obj" : ".o";
        llvm::sys::path::append(objectPath, name);

        auto objectPathStr = objectPath.str().str();
        return {objectPathStr, objectPathStr + ".deps"};
    }

    // the hash of a file's content, or empty if it cannot be read; once per file and run
    std::string fileHash(llvm::StringRef path)
    {
        auto it = fileHashes.find(path);
        if (it != fileHashes.end())
        {
            return it->second;
        }

        std::string hash;
        auto fileOrErr = llvm::MemoryBuffer::getFile(path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
        if (fileOrErr)
        {
            hash = hashString((*fileOrErr)->getBuffer());
        }

        fileHashes[path] = hash;
        return hash;
    }

    bool loadFromCache(llvm::StringRef objectPath, llvm::StringRef manifestPath, Manifest &manifest, JitUnit &unit)
    {
        auto manifestOrErr = llvm::MemoryBuffer::getFile(manifestPath, /*IsText=*/true);
        if (!manifestOrErr || !manifest.parse((*manifestOrErr)->getBuffer()))
        {
            manifest = Manifest();
            return false;
        }

        auto stale = [&](llvm::StringRef why) {
            LLVM_DEBUG(llvm::dbgs() << "JIT cache: " << objectPath << " is out of date: " << why << "\n";);
            (void)why;
            manifest = Manifest();
            return false;
        };

        if (compilerId.empty() || manifest.compiler != compilerId)
        {
            return stale("compiled by another tslang");
        }

        for (auto &[hash, path] : manifest.deps)
        {
            if (fileHash(path) != hash)
            {
                return stale(path);
            }
        }

        // Volatile: read, not mapped, so another run can still replace the file while this one runs
        auto objectOrErr = llvm::MemoryBuffer::getFile(objectPath, /*IsText=*/false, /*RequiresNullTerminator=*/false,
                                                       /*IsVolatile=*/true);
        if (!objectOrErr || hashString((*objectOrErr)->getBuffer()) != manifest.objectHash)
        {
            return stale("the object is missing or is not the one compiled");
        }

        LLVM_DEBUG(llvm::dbgs() << "JIT cache: using " << objectPath << "\n";);
        unit.object = std::move(*objectOrErr);
        return true;
    }

    void storeInCache(llvm::StringRef objectPath, llvm::StringRef manifestPath, const Manifest &manifest,
                      const llvm::MemoryBuffer &object)
    {
        // Without a compiler to tell one build from another, the object could outlive the build it
        // belongs to: better compiled again every time.
        if (compilerId.empty())
        {
            return;
        }

        // Nowhere to write it (a read-only folder, say) is not an error: the program still runs,
        // it is compiled again next time.
        if (llvm::sys::fs::create_directories(llvm::sys::path::parent_path(objectPath)) ||
            !writeFileAtomically(objectPath, object.getBuffer()) || !writeFileAtomically(manifestPath, manifest.str()))
        {
            LLVM_DEBUG(llvm::dbgs() << "JIT cache: could not write " << objectPath << "\n";);
            return;
        }

        LLVM_DEBUG(llvm::dbgs() << "JIT cache: wrote " << objectPath << "\n";);
    }

    int compile(llvm::StringRef path, bool isMain, Manifest &manifest, JitUnit &unit)
    {
        LLVM_DEBUG(llvm::dbgs() << "JIT cache: compiling " << path << "\n";);

        // An imported module is compiled as `--emit=obj` compiles a module linked beside the
        // program: no entry point, its top level run from its global constructors, and what it
        // imports declared only - those have objects of their own.
        auto options = compileOptions;
        if (!isMain)
        {
            options.isJit = false;
            options.generateEntryPoint = false;
        }

        std::unique_ptr<mlir::MLIRContext> context;
        std::unique_ptr<llvm::SourceMgr> sourceMgr;
        mlir::OwningOpRef<mlir::ModuleOp> module;
        auto generate = [&]() {
            options.sourceImports.clear();
            options.sharedLibraryImports.clear();
            options.importCycle = false;
            module = nullptr;
            context = createMLIRContext();
            sourceMgr = std::make_unique<llvm::SourceMgr>();
            return compileTypeScriptFileIntoMLIR(*context, path, *sourceMgr, module, options);
        };

        if (auto error = generate())
        {
            return error;
        }

        // A module in an import cycle cannot be compiled by itself when the other side of the
        // cycle has code at its top level - the same holds for `--emit=obj`. So a program whose
        // imports lead back to a file on the way is compiled as the JIT without the cache
        // compiles it: one module, the imported files included with their bodies, an object that
        // needs no other.
        if (isMain && options.importCycle)
        {
            LLVM_DEBUG(llvm::dbgs() << "JIT cache: " << path << " has an import cycle, compiled as one module\n";);
            options.jitCache = false;
            if (auto error = generate())
            {
                return error;
            }

            options.sourceImports.clear();
        }

        if (auto error = runMLIRPasses(*context, *sourceMgr, module, options))
        {
            return error;
        }

        manifest.compiler = compilerId;
        manifest.imports = options.sourceImports;
        manifest.hasGCtorsMethod = isMain && (*module).lookupSymbol(MLIR_GCTORS) != nullptr;

        // What it was compiled from: every source file read - itself, the files it references,
        // the modules it imports (their declarations are compiled into it), lib.d.ts - and the
        // libraries it imports, whose declarations it was compiled against.
        llvm::StringSet<> depPaths;
        auto addDep = [&](llvm::StringRef depPath, std::string hash) {
            if (!hash.empty() && depPaths.insert(depPath).second)
            {
                manifest.deps.push_back({hash, depPath.str()});
            }
        };

        for (unsigned id = 1; id <= sourceMgr->getNumBuffers(); id++)
        {
            auto *buffer = sourceMgr->getMemoryBuffer(id);
            llvm::SmallString<256> depPath(buffer->getBufferIdentifier());
            llvm::sys::fs::make_absolute(depPath);
            // only files: generated sources (a library's declarations, say) come from a file named below
            if (!llvm::sys::fs::is_regular_file(depPath))
            {
                continue;
            }

            // what was read is what it was compiled from, even if the file changed since
            auto hash = hashString(buffer->getBuffer());
            addDep(depPath, hash);
        }

        for (auto &library : options.sharedLibraryImports)
        {
            addDep(library, fileHash(library));
        }

        registerMLIRDialects(*module);

        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
        if (!llvmModule)
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "failed to emit LLVM IR\n";
            return -1;
        }

        llvmModule->setDataLayout(tm->createDataLayout());
        llvmModule->setTargetTriple(tm->getTargetTriple());

        auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel, options);
        if (auto err = optPipeline(llvmModule.get()))
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "failed to optimize LLVM IR, error: " << err << "\n";
            llvm::consumeError(std::move(err));
            return -1;
        }

        if (isMain)
        {
            // after the optimizer, so it cannot inline an entry with debug info into a thunk without any
            if (auto err = addEntryThunk(*llvmModule, mainFuncName.getValue()))
            {
                llvm::WithColor::error(llvm::errs(), "tslang") << err << "\n";
                llvm::consumeError(std::move(err));
                return -1;
            }
        }

        weakenComdatDefinitions(*llvmModule);
        manifest.initFunction = moveGlobalCtorsIntoFunction(*llvmModule, JIT_INIT_PREFIX + hashString(path));

        llvm::orc::SimpleCompiler compiler(*tm);
        auto objectOrErr = compiler(*llvmModule);
        if (!objectOrErr)
        {
            auto err = objectOrErr.takeError();
            llvm::WithColor::error(llvm::errs(), "tslang") << "failed to compile '" << path << "', error: " << err << "\n";
            llvm::consumeError(std::move(err));
            return -1;
        }

        unit.object = std::move(*objectOrErr);
        manifest.objectHash = hashString(unit.object->getBuffer());
        return 0;
    }

    const CompileOptions &compileOptions;
    llvm::orc::JITTargetMachineBuilder &tmBuilder;
    std::unique_ptr<llvm::TargetMachine> tm;
    std::vector<JitUnit> &units;
    std::string compilerId;
    llvm::StringSet<> visited;
    llvm::StringMap<std::string> fileHashes;
};

} // namespace

int buildJitUnits(const char *argv0, llvm::StringRef mainFile, const CompileOptions &compileOptions,
                  llvm::orc::JITTargetMachineBuilder &tmBuilder, std::vector<JitUnit> &units)
{
    JitUnitBuilder builder(argv0, compileOptions, tmBuilder, units);
    return builder.add(mainFile, /*isMain=*/true);
}
