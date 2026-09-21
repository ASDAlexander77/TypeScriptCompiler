#include "TypeScript/ObjDumper.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
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

// The bytes of a PE image from `rva` to the end of the section's data in the file. Only the part of
// a section that is in the file counts: past SizeOfRawData the loader zero-fills, and nothing a
// pointer could usefully point at is there.
static std::optional<ArrayRef<uint8_t>> imageBytesAt(const COFFObjectFile &coffObj, uint32_t rva)
{
    auto fileData = coffObj.getData();
    for (const SectionRef &sectionRef : coffObj.sections())
    {
        const coff_section *section = coffObj.getCOFFSection(sectionRef);
        uint32_t start = section->VirtualAddress;
        uint32_t size = section->VirtualSize ? std::min<uint32_t>(section->VirtualSize, section->SizeOfRawData)
                                             : section->SizeOfRawData;
        if (rva < start || rva - start >= size)
        {
            continue;
        }

        uint64_t offset = uint64_t(section->PointerToRawData) + (rva - start);
        uint64_t end = std::min<uint64_t>(uint64_t(section->PointerToRawData) + size, fileData.size());
        if (offset >= end)
        {
            return std::nullopt;
        }

        return ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(fileData.data()) + offset, end - offset);
    }

    return std::nullopt;
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

uint16_t coffMachine(StringRef filePath)
{
    auto expectedOwningBinary = createBinary(filePath);
    if (!expectedOwningBinary)
    {
        consumeError(expectedOwningBinary.takeError());
        return 0;
    }

    auto *binary = expectedOwningBinary.get().getBinary();
    if (auto *coffObj = dyn_cast<COFFObjectFile>(binary))
    {
        return coffObj->getMachine();
    }

    auto *archive = dyn_cast<Archive>(binary);
    if (!archive)
    {
        return 0;
    }

    // A static library holds objects; an import library holds short import members (and a few
    // objects for the import descriptors). Both carry the machine, the linker members before them
    // do not.
    Error err = Error::success();
    for (auto &child : archive->children(err))
    {
        auto childBinary = child.getAsBinary();
        if (!childBinary)
        {
            consumeError(childBinary.takeError());
            continue;
        }

        uint16_t machine = 0;
        if (auto *coffObj = dyn_cast<COFFObjectFile>(childBinary->get()))
        {
            machine = coffObj->getMachine();
        }
        else if (auto *importFile = dyn_cast<COFFImportFile>(childBinary->get()))
        {
            machine = importFile->getMachine();
        }

        // First non-zero, not first COFF member: machine-independent objects carry IMAGE_FILE_MACHINE_UNKNOWN (0).
        if (machine != 0)
        {
            consumeError(std::move(err));
            return machine;
        }
    }

    consumeError(std::move(err));
    return 0;
}

std::optional<std::string> readExportedCString(StringRef path, StringRef symbol)
{
    auto expectedOwningBinary = createBinary(path);
    if (!expectedOwningBinary)
    {
        consumeError(expectedOwningBinary.takeError());
        return std::nullopt;
    }

    auto *coffObj = dyn_cast<COFFObjectFile>(expectedOwningBinary.get().getBinary());
    if (!coffObj)
    {
        return std::nullopt;
    }

    // An image has one of the two optional headers; an object file has neither and no exports.
    auto pe32 = coffObj->getPE32Header() != nullptr;
    if (!pe32 && !coffObj->getPE32PlusHeader())
    {
        return std::nullopt;
    }

    for (const ExportDirectoryEntryRef &entry : coffObj->export_directories())
    {
        StringRef name;
        if (Error err = entry.getSymbolName(name))
        {
            consumeError(std::move(err));
            continue;
        }

        if (name != symbol)
        {
            continue;
        }

        bool forwarder = false;
        if (Error err = entry.isForwarder(forwarder))
        {
            consumeError(std::move(err));
            return std::nullopt;
        }

        if (forwarder)
        {
            return std::nullopt;
        }

        uint32_t slotRva = 0;
        if (Error err = entry.getExportRVA(slotRva))
        {
            consumeError(std::move(err));
            return std::nullopt;
        }

        // The exported variable is a pointer. The file holds its value as linked, which is the
        // string's address at the preferred image base; the loader's base relocation for the slot
        // would only add the load delta, so subtracting the preferred base gives the string's RVA.
        auto pointerSize = pe32 ? 4u : 8u;
        auto slot = imageBytesAt(*coffObj, slotRva);
        if (!slot || slot->size() < pointerSize)
        {
            return std::nullopt;
        }

        uint64_t address = pe32 ? uint64_t(support::endian::read32le(slot->data()))
                                : support::endian::read64le(slot->data());
        uint64_t imageBase = coffObj->getImageBase();
        if (address < imageBase || address - imageBase > UINT32_MAX)
        {
            return std::nullopt;
        }

        auto text = imageBytesAt(*coffObj, uint32_t(address - imageBase));
        if (!text)
        {
            return std::nullopt;
        }

        StringRef bytes(reinterpret_cast<const char *>(text->data()), text->size());
        auto nul = bytes.find('\0');
        if (nul == StringRef::npos)
        {
            return std::nullopt;
        }

        return bytes.take_front(nul).str();
    }

    return std::nullopt;
}

}
