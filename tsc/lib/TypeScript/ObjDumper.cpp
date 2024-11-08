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

void COFFDumper::getSymbols(SmallVector<StringRef> &symbols)
{
    export_directory_iterator I = coffObj.export_directory_begin();
    export_directory_iterator E = coffObj.export_directory_end();
    if (I == E)
    {
        return;
    }

    StringRef dllName;
    uint32_t ordinalBase;
    if (I->getDllName(dllName))
    {
        return;
    }

    if (I->getOrdinalBase(ordinalBase))
    {
        return;
    }

    for (; I != E; I = ++I)
    {
        uint32_t RVA;
        if (I->getExportRVA(RVA))
        {
            return;
        }

        StringRef name;
        if (I->getSymbolName(name))
        {
            continue;
        }

        if (!RVA && name.empty())
        {
            continue;
        }

        uint32_t ordinal;
        if (I->getOrdinal(ordinal))
        {
            return;
        }

        bool isForwarder;
        if (I->isForwarder(isForwarder))
        {
            return;
        }

        if (!name.empty())
        {
            symbols.push_back(name);
        }

        if (isForwarder)
        {
            StringRef s;
            if (I->getForwardTo(s))
            {
                return;
            }
        }
    }
}

Expected<std::unique_ptr<Dumper>> createDumper(const ObjectFile &objFile) 
{
    if (const auto *obj = dyn_cast<COFFObjectFile>(&objFile))
        return createCOFFDumper(*obj);
    // if (const auto *obj = dyn_cast<ELFObjectFileBase>(&objFile))
    //     return createELFDumper(*obj);
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

SmallVector<StringRef> getSymbols(StringRef filePath)
{
    SmallVector<StringRef> symbols{};

    auto expectedOwningBinary = createBinary(filePath);
    if (expectedOwningBinary)
    {
        auto &binary = *expectedOwningBinary.get().getBinary();
        if (auto *objFile = dyn_cast<ObjectFile>(&binary))
        {
            auto dumperOrErr = createDumper(*objFile);
            if (!dumperOrErr) {
                // report error
                return symbols;
            }

            auto &dumper = **dumperOrErr;
            dumper.getSymbols(symbols);
        }
    }

    return symbols;
}

}