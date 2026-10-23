// OwnershipSignaturePass (see OwnershipSignaturePass.h): a call resolved to its callees - class
// vtables and their families, imported declarations and the C-scalar surface (spec 31).

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    void OwnershipSignaturePass::collectClassVTables(mlir::ModuleOp module)
    {
        llvm::SmallVector<std::pair<llvm::StringRef, llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>>>> vtables;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto entries = classVTableEntries(globalOp);
            if (!entries.empty())
            {
                auto name = globalOp.getSymName();
                classNames.insert(name.drop_back(llvm::StringRef("..vtbl").size()));
                vtables.push_back({name, std::move(entries)});
            }
        });

        for (auto &[name, entries] : vtables)
        {
            for (auto [position, symbol] : entries)
            {
                families[{position, methodName(symbol.getValue())}].push_back(symbol);
            }
        }
    }

    mlir_ts::TupleType OwnershipSignaturePass::storageTupleOf(mlir::Type type)
    {
        return own_facts::storageTupleOf(&getContext(), type);
    }

    llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> OwnershipSignaturePass::classVTableEntries(mlir_ts::GlobalOp globalOp)
    {
        llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> entries;
        if (!globalOp.getSymName().ends_with("..vtbl"))
        {
            return entries;
        }

        auto isClass = false;
        globalOp.getInitializerRegion().walk([&](mlir_ts::InsertPropertyOp insertOp) {
            auto symbolRefOp = insertOp.getValue().getDefiningOp<mlir_ts::SymbolRefOp>();
            if (!symbolRefOp || insertOp.getPosition().size() != 1)
            {
                return;
            }

            auto symbol = symbolRefOp.getIdentifierAttr().getAttr();
            isClass = isClass || symbol.getValue().ends_with("..instanceOf");
            entries.push_back({insertOp.getPosition()[0], symbol});
        });

        if (!isClass)
        {
            entries.clear();
        }

        return entries;
    }

    bool OwnershipSignaturePass::isClassVTableEntry(mlir::Operation *op)
    {
        auto globalOp = op->getParentOfType<mlir_ts::GlobalOp>();
        return globalOp && !classVTableEntries(globalOp).empty();
    }

    std::string OwnershipSignaturePass::methodName(llvm::StringRef symbol)
    {
        size_t best = 0;
        for (auto &entry : classNames)
        {
            auto name = entry.getKey();
            if (name.size() > best && symbol.size() > name.size() && symbol.starts_with(name) &&
                symbol[name.size()] == '.')
            {
                best = name.size() + 1;
            }
        }

        return symbol.drop_front(best).str();
    }

    llvm::SmallVector<mlir::StringAttr, 2> OwnershipSignaturePass::familyOf(int64_t index, mlir::StringAttr identifier)
    {
        llvm::SmallVector<mlir::StringAttr, 2> members;
        if (auto found = families.find({index, methodName(identifier.getValue())}); found != families.end())
        {
            members = found->second;
        }

        if (!llvm::is_contained(members, identifier))
        {
            members.push_back(identifier);
        }

        return members;
    }

    bool OwnershipSignaturePass::addDefined(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || funcOp.isDeclaration())
        {
            return false;
        }

        if (!llvm::is_contained(callees.funcs, funcOp))
        {
            callees.funcs.push_back(funcOp);
        }

        return true;
    }

    bool OwnershipSignaturePass::addImported(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || !funcOp.isDeclaration() ||
            !(importedNoDrops.contains(symbol.getValue()) || funcOp->hasAttr(OWN_DECLARED_NO_DROPS_ATTR_NAME) ||
              (scalarDeclaration(funcOp) && !functionsEscape)))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = symbol.getValue();
        return true;
    }

    bool OwnershipSignaturePass::scalarType(mlir::Type type)
    {
        if (auto literalType = mlir::dyn_cast<mlir_ts::LiteralType>(type))
        {
            return scalarType(literalType.getElementType());
        }

        if (auto enumType = mlir::dyn_cast<mlir_ts::EnumType>(type))
        {
            return scalarType(enumType.getElementType());
        }

        return mlir::isa<mlir_ts::NumberType, mlir_ts::BooleanType, mlir_ts::ByteType, mlir_ts::CharType, mlir::IntegerType,
                         mlir::FloatType, mlir::IndexType>(type);
    }

    bool OwnershipSignaturePass::scalarSignature(mlir_ts::FunctionType functionType)
    {
        return !functionType.isVarArg() && llvm::all_of(functionType.getInputs(), scalarType) &&
               llvm::all_of(functionType.getResults(),
                            [](mlir::Type type) { return mlir::isa<mlir_ts::VoidType>(type) || scalarType(type); });
    }

    bool OwnershipSignaturePass::scalarDeclaration(mlir_ts::FuncOp funcOp)
    {
        return funcOp.isDeclaration() && !funcOp->hasAttr("export") && !funcOp->hasAttr("import") &&
               !funcOp->hasAttr(DLL_EXPORT) && !funcOp->hasAttr(DLL_IMPORT) && scalarSignature(funcOp.getFunctionType());
    }

    bool OwnershipSignaturePass::addImportedByAddress(mlir::Value callee, Callees &callees)
    {
        auto loadOp = callee.getDefiningOp<mlir_ts::LoadOp>();
        auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
        if (!addressOfOp)
        {
            return false;
        }

        auto globalOp = globals.lookup(addressOfOp.getGlobalName());
        if (!globalOp || writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            return false;
        }

        // the global is never anything but that address: `Cast(SearchForAddressOfSymbol(name))`
        auto &region = globalOp.getInitializerRegion();
        if (!region.hasOneBlock())
        {
            return false;
        }

        auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(region.front().getTerminator());
        auto castOp = resultOp && resultOp->getNumOperands() == 1
                          ? resultOp->getOperand(0).getDefiningOp<mlir_ts::CastOp>()
                          : mlir_ts::CastOp();
        auto searchOp = castOp ? castOp.getIn().getDefiningOp<mlir_ts::SearchForAddressOfSymbolOp>()
                               : mlir_ts::SearchForAddressOfSymbolOp();
        auto nameOp = searchOp ? searchOp->getOperand(0).getDefiningOp<mlir_ts::ConstantOp>() : mlir_ts::ConstantOp();
        auto name = nameOp ? mlir::dyn_cast<mlir::StringAttr>(nameOp.getValue()) : mlir::StringAttr();
        if (!name || !importedNoDrops.contains(name.getValue()))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = name.getValue();
        return true;
    }

    OwnershipSignaturePass::Callees OwnershipSignaturePass::resolve(mlir::Operation *op)
    {
        Callees callees;
        if (auto callee = directCallee(op))
        {
            callees.named.push_back(callee.getAttr());
            callees.known = addDefined(callee.getAttr(), callees) || addImported(callee.getAttr(), callees);
        }
        else if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(op))
        {
            resolveValue(calleeValue(op), callees);
        }

        if (!callees.known)
        {
            callees.funcs.clear();
        }

        return callees;
    }

    void OwnershipSignaturePass::resolveValue(mlir::Value callee, Callees &callees)
    {
        // `ts.Load(ts.AddressOf @g)`: a dynamic import's global, which may say what the function
        // keeps; a method of a class imported so is bound to its object first
        auto function = callee;
        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>();
        if (auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
            addressOfOp && !writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            callees.named.push_back(addressOfOp.getGlobalNameAttr().getAttr());
        }

        if (addImportedByAddress(callee, callees))
        {
            callees.known = true;
            return;
        }

        if (auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            callee = getMethodOp.getBoundFunc();
        }

        auto *def = callee.getDefiningOp();
        mlir::StringAttr identifier;
        int64_t index = -1;
        if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::SymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (isInstanceOfSlot(def))
        {
            callees.known = true;
            callees.instanceOf = true;
            return;
        }
        else if (auto symbol = fieldFunctions.of(def))
        {
            callees.known = addDefined(symbol, callees);
            callees.fieldFunction = callees.known;
            return;
        }
        else if (auto interfaceRefOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(def))
        {
            resolveInterfaceSlot(interfaceRefOp, interfaceFamilies, callees);
            return;
        }
        else if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def))
        {
            resolveCallback(loadOp, callees);
            return;
        }
        else if (auto boundOp = mlir::dyn_cast_or_null<mlir_ts::CreateBoundFunctionOp>(def))
        {
            // a function-typed field of an interface, called: `ts.CreateBoundFunction(
            // ts.ExtractInterfaceThis(i), ts.Load(ts.InterfaceSymbolRef(i, f)))`
            auto loadOp = boundOp.getFunc().getDefiningOp<mlir_ts::LoadOp>();
            auto refOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>() : mlir_ts::InterfaceSymbolRefOp();
            auto thisOp = boundOp.getThisVal().getDefiningOp<mlir_ts::ExtractInterfaceThisOp>();
            if (refOp && isInterfaceField(refOp.getResult()) && thisOp && thisOp.getInterfaceVal() == refOp.getInterfaceVal())
            {
                resolveInterfaceSlot(refOp, interfaceFieldFamilies, callees);
            }

            return;
        }

        if (!identifier)
        {
            return;
        }

        if (index < 0)
        {
            callees.named.push_back(identifier);
            callees.known = addDefined(identifier, callees) || addImported(identifier, callees);
            return;
        }

        callees.named = familyOf(index, identifier);
        resolveFamily(index, identifier, callees);
    }

    void OwnershipSignaturePass::resolveFamily(int64_t index, mlir::StringAttr identifier, Callees &callees)
    {
        callees.known = true;
        for (auto member : familyOf(index, identifier))
        {
            callees.known = addDefined(member, callees) && callees.known;
        }

        callees.known = callees.known && llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp funcOp) { return !callableFromOutside(funcOp); });
    }

    bool OwnershipSignaturePass::isInstanceOfSlot(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        auto slot = castOp ? castOp.getIn().getDefiningOp<mlir_ts::VTableOffsetRefOp>() : mlir_ts::VTableOffsetRefOp();
        if (!slot)
        {
            return false;
        }

        // a class vtable, read out of an object's first word, whose slot 0 is the class's
        // `.instanceOf` (CLASS_VTABLE_INSTANCEOF_SLOT), or an interface's own vtable, whose header
        // holds its implementer's (INTERFACE_VTABLE_INSTANCEOF_SLOT)
        auto *vtable = slot.getVtable().getDefiningOp();
        if (auto classVTable = mlir::dyn_cast_or_null<mlir_ts::VTableOffsetRefOp>(vtable))
        {
            return classVTable.getIndex() == 0 && slot.getIndex() == CLASS_VTABLE_INSTANCEOF_SLOT;
        }

        return mlir::isa_and_nonnull<mlir_ts::ExtractInterfaceVTableOp>(vtable) &&
               slot.getIndex() == INTERFACE_VTABLE_INSTANCEOF_SLOT;
    }

} // namespace own_signature
