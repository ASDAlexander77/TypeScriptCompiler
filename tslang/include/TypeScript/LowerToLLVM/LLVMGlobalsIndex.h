#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMGLOBALSINDEX_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMGLOBALSINDEX_H_

#include "TypeScript/DataStructs.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <system_error>

namespace typescript
{

// The LLVM globals of the module being lowered: which names are taken, and which global is the
// last one - of any kind, and of those holding a string or dense elements - at the top level of
// the module.
//
// Lowering asks both questions for every global it makes ("is it there yet?", then "where does
// the next one go?"), and answering each by scanning the module's operations made the lowering
// of a module with thousands of classes quadratic. The index answers them from a table, and
// gives the answers the scans gave: the same global, the same insertion point.
//
// It belongs to one run of the LLVM lowering pass, which puts it in
// CompileOptions::llvmGlobalsIndex for the conversion and takes it out after. Two rules keep it
// true:
//  - every LLVM global made during the conversion is reported with created() (the helpers'
//    createGlobal does it). One that is not reported and is not where the rules of created()
//    can place it would be missed by lookup();
//  - the conversion driver erases what a failed pattern created. The index is the conversion's
//    listener, and forgets everything when a pattern that reported a global fails; the next
//    question rebuilds it with one scan.
//
// Set TSLANG_VERIFY_LLVM_GLOBALS_INDEX to a file name to have every answer checked against the
// scan it replaces: a difference is appended to the file, and the scan's answer is used.
class LLVMGlobalsIndex : public mlir::RewriterBase::Listener
{
  public:
    enum class Kind
    {
        Any,
        String,
        Dense,
        Count
    };

    explicit LLVMGlobalsIndex(mlir::ModuleOp module) : module(module)
    {
        if (auto *file = std::getenv("TSLANG_VERIFY_LLVM_GLOBALS_INDEX"))
        {
            verifyFile = file;
        }
    }

    // the index of the conversion under way, if any
    static LLVMGlobalsIndex *of(CompileOptions &compileOptions)
    {
        return static_cast<LLVMGlobalsIndex *>(compileOptions.llvmGlobalsIndex);
    }

    bool isFor(mlir::ModuleOp otherModule)
    {
        return otherModule == module;
    }

    bool isFor(mlir::Block *block)
    {
        return block == module.getBody();
    }

    // module.lookupSymbol<LLVM::GlobalOp>(name)
    mlir::LLVM::GlobalOp lookup(llvm::StringRef name)
    {
        build();

        auto it = byName.find(name);
        auto *found = it == byName.end() ? nullptr : it->second;
        if (!verifyFile.empty())
        {
            auto scanned = module.lookupSymbol<mlir::LLVM::GlobalOp>(name);
            if (scanned.getOperation() != found)
            {
                report("lookup", name);
                return scanned;
            }
        }

        return found ? llvm::cast<mlir::LLVM::GlobalOp>(found) : mlir::LLVM::GlobalOp();
    }

    // module.lookupSymbol(name): the first operation of the name at the top level of the module.
    //
    // The lookup scans the module, and lowering asks it for the callee of every call. Here the
    // symbols the module has before the conversion are tabled (tableSymbols), and the answer is
    // found from the tabled one: an operation that replaces a symbol during the conversion (a ts
    // function by a func.func, that by an llvm.func) is created in front of the one it replaces,
    // which stays where it was until the conversion ends, so the first of the name is reached by
    // stepping back over the operations of that name. The conversion never erases those symbols
    // before it ends.
    //
    // A symbol made during the conversion (a helper routine, a global, a comdat table) is in a
    // second table, which createdSymbol fills; what a failed pattern created is erased, so that
    // table is made again by a scan after one fails.
    //
    // A name in neither table is looked up by the scan, unless it is the lowering's own
    // (`ownName`): a name only tslang's lowering gives a symbol - a release routine of a type,
    // "__mlir_gctors". Such a name in neither table is not in the module. The helpers ask "is this
    // routine there yet?" for every type, the answer is nearly always no, and a scan to learn it
    // read the whole module each time. It holds because every symbol the lowering makes is
    // reported (createSymbol, the helpers' createGlobal). It would not hold for any name: MLIR's
    // own patterns make symbols too (the lowering of cf.assert declares "puts").
    void tableSymbols()
    {
        for (auto &moduleOp : *module.getBody())
        {
            if (auto symbolName = symbolNameOf(&moduleOp))
            {
                symbols.try_emplace(symbolName, &moduleOp);
            }
        }
    }

    // a symbol the lowering made at the top level of the module
    void createdSymbol(mlir::Operation *op)
    {
        created_++;
        noteCreatedSymbol(op);
    }

    // module.lookupSymbol(name), from the index while the module is being converted
    static mlir::Operation *lookupSymbolIn(CompileOptions &compileOptions, mlir::ModuleOp module, llvm::StringRef name,
                                           bool ownName = false)
    {
        auto *index = of(compileOptions);
        if (index && index->isFor(module))
        {
            return index->lookupSymbol(name, ownName);
        }

        return module.lookupSymbol(name);
    }

    // module.lookupSymbol<T>(name)
    template <typename T>
    static T lookupSymbolAs(CompileOptions &compileOptions, mlir::ModuleOp module, llvm::StringRef name)
    {
        return llvm::dyn_cast_or_null<T>(lookupSymbolIn(compileOptions, module, name));
    }

    // module.lookupSymbol<T>(name) for a name that only tslang's lowering gives a symbol
    template <typename T>
    static T lookupOwnSymbolAs(CompileOptions &compileOptions, mlir::ModuleOp module, llvm::StringRef name)
    {
        return llvm::dyn_cast_or_null<T>(lookupSymbolIn(compileOptions, module, name, /*ownName=*/true));
    }

    // builder.create<OpTy>(...) for a symbol at the top level of the module, told to the index
    template <typename OpTy, typename... Args>
    static OpTy createSymbol(CompileOptions &compileOptions, mlir::OpBuilder &builder, Args &&...args)
    {
        auto symbolOp = builder.create<OpTy>(std::forward<Args>(args)...);
        if (auto *index = of(compileOptions))
        {
            index->createdSymbol(symbolOp);
        }

        return symbolOp;
    }

    mlir::Operation *lookupSymbol(llvm::StringRef name, bool ownName = false)
    {
        auto nameAttr = mlir::StringAttr::get(module.getContext(), name);
        mlir::Operation *found = nullptr;
        auto it = symbols.find(nameAttr);
        if (it != symbols.end())
        {
            found = it->second;
        }
        else
        {
            if (!createdSymbolsKnown)
            {
                createdSymbols.clear();
                for (auto &moduleOp : *module.getBody())
                {
                    auto symbolName = symbolNameOf(&moduleOp);
                    if (symbolName && !symbols.contains(symbolName))
                    {
                        createdSymbols.try_emplace(symbolName, &moduleOp);
                    }
                }

                createdSymbolsKnown = true;
            }

            auto created = createdSymbols.find(nameAttr);
            if (created != createdSymbols.end())
            {
                found = created->second;
            }
            else if (!ownName)
            {
                return module.lookupSymbol(name);
            }
        }

        if (found)
        {
            for (auto *previous = found->getPrevNode(); previous && symbolNameOf(previous) == nameAttr;
                 previous = previous->getPrevNode())
            {
                found = previous;
            }
        }

        if (!verifyFile.empty())
        {
            auto *scanned = module.lookupSymbol(name);
            if (scanned != found)
            {
                report("symbol lookup", name);
                return scanned;
            }
        }

        return found;
    }

    // The comdat table that takes the next selector, with how many it holds and its number (the
    // module's tables are numbered from 0 as they are made): null when the module has none. The
    // lowering keeps a table small (COMDAT_TABLE_SELECTORS in LowerToLLVM.cpp) and starts another.
    mlir::Operation *comdatTable(unsigned &selectors, unsigned &number)
    {
        if (!comdatKnown)
        {
            comdat = nullptr;
            comdatSelectors = 0;
            comdatNumber = 0;
            for (auto &moduleOp : *module.getBody())
            {
                auto comdatOp = llvm::dyn_cast<mlir::LLVM::ComdatOp>(&moduleOp);
                if (!comdatOp)
                {
                    continue;
                }

                // the newest is the one with the highest number, which its name ends with
                unsigned tableNumber = 0;
                auto name = comdatOp.getSymName();
                auto underscore = name.rfind('_');
                if (underscore == llvm::StringRef::npos || name.substr(underscore + 1).getAsInteger(10, tableNumber))
                {
                    // the first one's name has no number
                    tableNumber = 0;
                }

                if (!comdat || tableNumber >= comdatNumber)
                {
                    comdat = &moduleOp;
                    comdatNumber = tableNumber;
                    comdatSelectors = static_cast<unsigned>(
                        std::distance(comdatOp.getBody().front().begin(), comdatOp.getBody().front().end()));
                }
            }

            comdatKnown = true;
        }

        selectors = comdatSelectors;
        number = comdatNumber;
        return comdat;
    }

    void comdatTableCreated(mlir::Operation *table, unsigned number)
    {
        created_++;
        noteCreatedSymbol(table);
        comdat = table;
        comdatSelectors = 0;
        comdatNumber = number;
        comdatKnown = true;
    }

    void comdatSelectorAdded()
    {
        created_++;
        comdatSelectors++;
    }

    // the last global of the kind at the top level of the module, or null
    mlir::Operation *last(Kind kind)
    {
        build();

        auto *found = lastOf[static_cast<int>(kind)];
        if (!verifyFile.empty())
        {
            auto *scanned = scanLast(kind);
            if (scanned != found)
            {
                report("last", kind == Kind::Any ? "any" : kind == Kind::String ? "string" : "dense");
                return scanned;
            }
        }

        return found;
    }

    // call right after the global is created, before anything else is inserted beside it
    void created(mlir::LLVM::GlobalOp globalOp)
    {
        created_++;
        noteCreatedSymbol(globalOp);

        if (!built || globalOp->getParentOp() != module.getOperation())
        {
            return;
        }

        if (!byName.try_emplace(globalOp.getSymName(), globalOp.getOperation()).second)
        {
            // a second symbol of the name: the scan's first-one-wins is what a rebuild gives
            built = false;
            return;
        }

        // Where it stands among the others is read off the operation before it, never asked of
        // the block's order: inserting again and again at one point invalidates that order, and
        // each question would then renumber the block.
        auto kind = kindOf(globalOp);
        auto *previous = globalOp->getPrevNode();
        auto &lastAny = lastOf[static_cast<int>(Kind::Any)];
        if (previous == lastAny)
        {
            // after the last global (or the first operation of a module with no global): the
            // last of all now, so the last of its kind too
            lastAny = globalOp;
            if (kind != Kind::Any)
            {
                lastOf[static_cast<int>(kind)] = globalOp;
            }
        }
        else if (!previous)
        {
            // the first operation of the module, in front of every other global
            if (kind != Kind::Any && !lastOf[static_cast<int>(kind)])
            {
                lastOf[static_cast<int>(kind)] = globalOp;
            }
        }
        else if (kind != Kind::Any && previous == lastOf[static_cast<int>(kind)])
        {
            // after the last of its kind, which is not the last of all: still in front of that one
            lastOf[static_cast<int>(kind)] = globalOp;
        }
        else
        {
            // somewhere else: only a scan can tell
            built = false;
        }
    }

    void notifyPatternBegin(const mlir::Pattern &, mlir::Operation *) override
    {
        createdAtPatternBegin.push_back(created_);
    }

    void notifyPatternEnd(const mlir::Pattern &, mlir::LogicalResult status) override
    {
        auto createdBefore = createdAtPatternBegin.empty() ? 0 : createdAtPatternBegin.pop_back_val();
        if (mlir::failed(status) && createdBefore != created_)
        {
            // what the pattern created, and the patterns applied to what it created, is erased
            // (or about to be): no pointer taken since it began can be trusted
            built = false;
            comdatKnown = false;
            createdSymbolsKnown = false;
        }
    }

  private:
    static mlir::StringAttr symbolNameOf(mlir::Operation *op)
    {
        return op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());
    }

    void noteCreatedSymbol(mlir::Operation *op)
    {
        if (!createdSymbolsKnown || op->getParentOp() != module.getOperation())
        {
            return;
        }

        auto symbolName = symbolNameOf(op);
        if (symbolName && !symbols.contains(symbolName))
        {
            createdSymbols.try_emplace(symbolName, op);
        }
    }

    static Kind kindOf(mlir::LLVM::GlobalOp globalOp)
    {
        auto value = globalOp.getValueAttr();
        if (value && llvm::isa<mlir::StringAttr>(value))
        {
            return Kind::String;
        }

        if (value && llvm::isa<mlir::DenseElementsAttr>(value))
        {
            return Kind::Dense;
        }

        return Kind::Any;
    }

    void build()
    {
        if (built)
        {
            return;
        }

        byName.clear();
        for (auto &last : lastOf)
        {
            last = nullptr;
        }

        for (auto &moduleOp : *module.getBody())
        {
            if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(&moduleOp))
            {
                byName.try_emplace(globalOp.getSymName(), &moduleOp);
                lastOf[static_cast<int>(Kind::Any)] = &moduleOp;
                auto kind = kindOf(globalOp);
                if (kind != Kind::Any)
                {
                    lastOf[static_cast<int>(kind)] = &moduleOp;
                }
            }
        }

        built = true;
    }

    mlir::Operation *scanLast(Kind kind)
    {
        for (auto &moduleOp : llvm::reverse(*module.getBody()))
        {
            if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(&moduleOp))
            {
                if (kind == Kind::Any || kindOf(globalOp) == kind)
                {
                    return &moduleOp;
                }
            }
        }

        return nullptr;
    }

    void report(llvm::StringRef question, llvm::StringRef subject)
    {
        std::error_code ec;
        llvm::raw_fd_ostream out(verifyFile, ec, llvm::sys::fs::OF_Append);
        if (!ec)
        {
            out << "LLVM globals index: " << question << " '" << subject << "' differs from the scan\n";
        }
    }

    mlir::ModuleOp module;
    // the first operation of each symbol name before the conversion (tableSymbols)
    llvm::DenseMap<mlir::StringAttr, mlir::Operation *> symbols;
    // ...and of each name made since (createdSymbol); empty and known as the conversion begins
    bool createdSymbolsKnown = true;
    llvm::DenseMap<mlir::StringAttr, mlir::Operation *> createdSymbols;
    // the comdat table taking selectors (comdatTable)
    bool comdatKnown = false;
    mlir::Operation *comdat = nullptr;
    unsigned comdatSelectors = 0;
    unsigned comdatNumber = 0;
    bool built = false;
    llvm::StringMap<mlir::Operation *> byName;
    mlir::Operation *lastOf[static_cast<int>(Kind::Count)] = {};
    // globals reported so far, and the count when each pattern now being applied began
    unsigned created_ = 0;
    llvm::SmallVector<unsigned, 8> createdAtPatternBegin;
    std::string verifyFile;
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMGLOBALSINDEX_H_
