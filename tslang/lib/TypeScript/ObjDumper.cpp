#include "TypeScript/ObjDumper.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "obj"

using namespace llvm;
using namespace llvm::object;

std::unique_ptr<Dumper> createCOFFDumper(const object::COFFObjectFile &obj) {
  return std::make_unique<COFFDumper>(obj);
}

std::unique_ptr<Dumper> createELFDumper(const ELFObjectFileBase &elfObjectFile)
{
    if (const auto *o = dyn_cast<ELF32LEObjectFile>(&elfObjectFile))
        return createDumperT(*o);
    if (const auto *o = dyn_cast<ELF32BEObjectFile>(&elfObjectFile))
        return createDumperT(*o);
    if (const auto *o = dyn_cast<ELF64LEObjectFile>(&elfObjectFile))
        return createDumperT(*o);
    return createDumperT(cast<ELF64BEObjectFile>(elfObjectFile));
}

void COFFDumper::getSymbols(SmallVector<StringRef> &symbols, BumpPtrAllocator &stringAllocator)
{
    export_directory_iterator I = coffObj.export_directory_begin();
    export_directory_iterator E = coffObj.export_directory_end();
    if (I == E)
    {
        return;
    }

    for (; I != E; I = ++I)
    {
        StringRef name;
        if (I->getSymbolName(name))
        {
            continue;
        }

        if (!name.empty())
        {
            symbols.push_back(StringRef(name).copy(stringAllocator));
        }
    }
}

Expected<std::unique_ptr<Dumper>> createDumper(const ObjectFile &objFile) 
{
    if (const auto *obj = dyn_cast<COFFObjectFile>(&objFile))
        return createCOFFDumper(*obj);
    if (const auto *obj = dyn_cast<ELFObjectFileBase>(&objFile))
        return createELFDumper(*obj);
    // if (const auto *obj = dyn_cast<MachOObjectFile>(&objFile))
    //     return createMachODumper(*obj);
    // if (const auto *obj = dyn_cast<WasmObjectFile>(&objFile))
    //     return createWasmDumper(*obj);
    // if (const auto *obj = dyn_cast<XCOFFObjectFile>(&objFile))
    //     return createXCOFFDumper(*obj);

    return createStringError(errc::invalid_argument,
                            "unsupported object file format");
}

namespace Dump
{

void getSymbols(StringRef filePath, SmallVector<StringRef> &symbols, llvm::BumpPtrAllocator &stringAllocator)
{
    auto expectedOwningBinary = createBinary(filePath);
    if (expectedOwningBinary)
    {
        auto &binary = *expectedOwningBinary.get().getBinary();
        if (auto *objFile = dyn_cast<ObjectFile>(&binary))
        {
            auto dumperOrErr = createDumper(*objFile);
            if (!dumperOrErr) {
                return;
            }

            auto &dumper = **dumperOrErr;
            dumper.getSymbols(symbols, stringAllocator);
        }
    }
}

bool containsGarbageCollector(StringRef filePath)
{
    // Boehm's GC_init reads this environment variable, so its name is a string constant in every
    // binary that contains the collector, in every build flavour, stripped or not - and in none
    // that only calls it. Its internal symbols are no help: a DLL does not export them.
    static const StringRef fingerprint = "GC_INITIAL_HEAP_SIZE";

    auto expectedOwningBinary = createBinary(filePath);
    if (!expectedOwningBinary)
    {
        consumeError(expectedOwningBinary.takeError());
        return false;
    }

    auto *objFile = dyn_cast<ObjectFile>(expectedOwningBinary.get().getBinary());
    if (!objFile)
    {
        return false;
    }

    for (const SectionRef &section : objFile->sections())
    {
        if (!section.isData() || section.isBSS())
        {
            continue;
        }

        auto contents = section.getContents();
        if (!contents)
        {
            consumeError(contents.takeError());
            continue;
        }

        if (contents->contains(fingerprint))
        {
            return true;
        }
    }

    return false;
}

}