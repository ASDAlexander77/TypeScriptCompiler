// A module's exported declarations, as an importer reads them from the file beside a shared library
// (TypeScript/LibraryDeclarations.h). MLIRGen and the passes keep them in the module as exported
// globals - `__decls_<module>`, `__decls_generic_<module>`, the memory model marker
// `__tsmm_<model>_<module>` and, under own, `__tsown_<module>` - since the ownership signature
// pass reads the first and writes the last. Once the module is LLVM IR they are taken out of it:
// written beside the object or the library, and never part of the binary.

#include "TypeScript/Defines.h"
#include "TypeScript/LibraryDeclarations.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <map>

using namespace typescript;

// the NUL-terminated text a global holding a pointer into a constant string points at
static std::optional<std::string> pointedText(llvm::GlobalVariable &global)
{
    if (!global.hasInitializer())
    {
        return std::nullopt;
    }

    auto &dataLayout = global.getParent()->getDataLayout();
    llvm::APInt offset(dataLayout.getIndexTypeSizeInBits(global.getType()), 0);
    auto *base = global.getInitializer()->stripAndAccumulateConstantOffsets(dataLayout, offset, true);
    auto *stringGlobal = llvm::dyn_cast_or_null<llvm::GlobalVariable>(base);
    if (!stringGlobal || !stringGlobal->hasInitializer())
    {
        return std::nullopt;
    }

    auto *data = llvm::dyn_cast<llvm::ConstantDataSequential>(stringGlobal->getInitializer());
    if (!data)
    {
        return std::nullopt;
    }

    auto bytes = data->getRawDataValues();
    if (offset.getZExtValue() > bytes.size())
    {
        return std::nullopt;
    }

    bytes = bytes.drop_front(offset.getZExtValue());
    return bytes.take_until([](char c) { return c == '\0'; }).str();
}

std::string takeLibraryDeclarations(llvm::Module &module, llvm::StringRef memoryModel)
{
    const llvm::StringRef declsPrefix = SHARED_LIB_DECLARATIONS_2UNDERSCORE "_";
    const llvm::StringRef genericPrefix = SHARED_LIB_DECLARATIONS_2UNDERSCORE "_generic_";

    std::vector<LibraryModuleDeclarations> modules;
    std::map<std::string, size_t> indexOf;
    auto moduleFor = [&](llvm::StringRef suffix) -> LibraryModuleDeclarations & {
        auto [it, inserted] = indexOf.insert({suffix.str(), modules.size()});
        if (inserted)
        {
            modules.push_back({});
            modules.back().suffix = suffix.str();
        }

        return modules[it->second];
    };

    llvm::SmallVector<llvm::GlobalVariable *> taken;
    for (auto &global : module.globals())
    {
        auto name = global.getName();
        if (name.starts_with(genericPrefix))
        {
            if (auto text = pointedText(global))
            {
                moduleFor(name.drop_front(genericPrefix.size())).genericDeclarations = *text;
            }
        }
        else if (name.starts_with(declsPrefix))
        {
            if (auto text = pointedText(global))
            {
                moduleFor(name.drop_front(declsPrefix.size())).declarations = *text;
            }
        }
        else if (name.starts_with(SHARED_LIB_MEMORY_MODEL))
        {
            // "__tsmm_<model>_<module>": the model is the segment after the prefix
            auto rest = name.drop_front(llvm::StringRef(SHARED_LIB_MEMORY_MODEL).size());
            auto [model, suffix] = rest.split('_');
            moduleFor(suffix).memoryModel = model.str();
        }
        else if (name.starts_with(SHARED_LIB_OWN_FACTS))
        {
            if (auto text = pointedText(global))
            {
                moduleFor(name.drop_front(llvm::StringRef(SHARED_LIB_OWN_FACTS).size())).ownFacts = *text;
            }
        }
        else
        {
            continue;
        }

        taken.push_back(&global);
    }

    auto *gctors = module.getFunction(MLIR_GCTORS);
    auto hasGCtors = gctors && !gctors->isDeclaration();
    for (auto &declarations : modules)
    {
        if (declarations.memoryModel.empty())
        {
            declarations.memoryModel = memoryModel.str();
        }

        declarations.hasGCtors = hasGCtors;
    }

    // a module with a marker but nothing to import (no exports) is no block of the file
    llvm::erase_if(modules, [](const LibraryModuleDeclarations &declarations) {
        return declarations.declarations.empty() && declarations.genericDeclarations.empty();
    });

    llvm::SmallPtrSet<llvm::GlobalValue *, 8> takenSet(taken.begin(), taken.end());
    llvm::removeFromUsedLists(module, [&](llvm::Constant *c) {
        return takenSet.contains(llvm::dyn_cast<llvm::GlobalValue>(c->stripPointerCasts()));
    });

    for (auto *global : taken)
    {
        if (!global->use_empty())
        {
            continue;
        }

        llvm::GlobalVariable *text = nullptr;
        if (global->hasInitializer())
        {
            llvm::APInt offset(module.getDataLayout().getIndexTypeSizeInBits(global->getType()), 0);
            text = llvm::dyn_cast_or_null<llvm::GlobalVariable>(
                global->getInitializer()->stripAndAccumulateConstantOffsets(module.getDataLayout(), offset, true));
        }

        global->eraseFromParent();

        // the text, when nothing else reads it (the same string elsewhere is the same constant)
        if (text && text->hasLocalLinkage())
        {
            text->removeDeadConstantUsers();
            if (text->use_empty())
            {
                text->eraseFromParent();
            }
        }
    }

    return modules.empty() ? std::string() : printLibraryDeclarations(modules);
}

// Writes the declarations beside a binary; with none, removes a file of this kind left there by an
// earlier build (never one somebody wrote).
int writeLibraryDeclarations(llvm::StringRef binaryPath, llvm::StringRef text)
{
    auto path = libraryDeclarationsPath(binaryPath);
    if (text.empty())
    {
        if (auto existing = llvm::MemoryBuffer::getFile(path); existing && isLibraryDeclarationsText((*existing)->getBuffer()))
        {
            llvm::sys::fs::remove(path);
        }

        return 0;
    }

    std::error_code ec;
    llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "cannot write '" << path << "': " << ec.message() << "\n";
        return -1;
    }

    out << text;
    return 0;
}

// the declarations written beside the objects a library is linked from (--emit=obj)
std::string readObjectDeclarations(llvm::StringRef objectPath)
{
    auto buffer = llvm::MemoryBuffer::getFile(libraryDeclarationsPath(objectPath));
    if (!buffer || !isLibraryDeclarationsText((*buffer)->getBuffer()))
    {
        return std::string();
    }

    return (*buffer)->getBuffer().str();
}
