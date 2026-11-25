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

        // each slot's members by the root of the vtable's class; a slot one of whose members has an
        // unknown root is one family, as every slot was before
        std::map<std::pair<int64_t, std::string>, llvm::SmallVector<std::pair<std::string, mlir::StringAttr>>> slots;
        for (auto &[name, entries] : vtables)
        {
            auto root = familyRoot(name.drop_back(llvm::StringRef("..vtbl").size()));
            for (auto [position, symbol] : entries)
            {
                slots[{position, methodName(symbol.getValue())}].push_back({root, symbol});
            }
        }

        for (auto &[slot, members] : slots)
        {
            auto unknownRoot = llvm::any_of(members, [](auto &member) { return member.first.empty(); });
            for (auto &[root, symbol] : members)
            {
                families[{slot.first, slot.second, unknownRoot ? std::string() : root}].push_back(symbol);
            }
        }
    }

    std::string OwnershipSignaturePass::familyRoot(llvm::StringRef className)
    {
        auto storage = classStorages.lookup(className);
        for (auto depth = 0; storage && depth < 64; ++depth)
        {
            auto fields = storage.getFields();
            auto base = fields.empty() ? mlir_ts::ClassStorageType() : mlir::dyn_cast<mlir_ts::ClassStorageType>(fields.front().type);
            if (!base)
            {
                return storage.getName().getValue().str();
            }

            storage = base;
        }

        return std::string();
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
        // the family of the identifier's class's root, or the slot's one family of an unknown root
        llvm::SmallVector<mlir::StringAttr, 2> members;
        auto name = methodName(identifier.getValue());
        auto className =
            name.size() < identifier.getValue().size() ? identifier.getValue().drop_back(name.size() + 1) : llvm::StringRef();
        auto found = families.find({index, name, familyRoot(className)});
        if (found == families.end())
        {
            found = families.find({index, name, std::string()});
        }

        if (found != families.end())
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
        if (!funcOp || !funcOp.isDeclaration())
        {
            return false;
        }

        if (!(importedNoDrops.contains(symbol.getValue()) || funcOp->hasAttr(OWN_DECLARED_NO_DROPS_ATTR_NAME) ||
              (scalarDeclaration(funcOp) && !functionsEscape)))
        {
            // a function of an imported module that overwrites only these places (--own-facts)
            auto found = importedFieldDrops.find(symbol.getValue());
            if (found == importedFieldDrops.end())
            {
                return false;
            }

            callees.importedOverwrites = true;
            callees.importedPlaces.assign(found->second.places.begin(), found->second.places.end());
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

    bool OwnershipSignaturePass::foreignOnlyDeclaration(mlir_ts::FuncOp funcOp)
    {
        // a number, a string, a `@cpp` handle, or one of them or null: nothing that reaches an object
        // of this module
        std::function<bool(mlir::Type)> foreign = [&](mlir::Type type) {
            if (scalarType(type) || mlir::isa<mlir_ts::StringType, mlir_ts::NullType, mlir_ts::UndefinedType, mlir_ts::VoidType>(type))
            {
                return true;
            }

            if (auto classType = mlir::dyn_cast<mlir_ts::ClassType>(type))
            {
                auto storage = mlir::dyn_cast<mlir_ts::ClassStorageType>(classType.getStorageType());
                return storage && storage.isCpp();
            }

            if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
            {
                return foreign(optionalType.getElementType());
            }

            if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
            {
                return llvm::all_of(unionType.getTypes(), foreign);
            }

            return false;
        };

        auto functionType = funcOp.getFunctionType();
        return funcOp.isDeclaration() && funcOp->hasAttr(OWN_DECLARED_NO_DROPS_ATTR_NAME) && !funcOp->hasAttr("export") &&
               !funcOp->hasAttr("import") && !funcOp->hasAttr(DLL_IMPORT) && !functionType.isVarArg() &&
               llvm::all_of(functionType.getInputs(), foreign) && llvm::all_of(functionType.getResults(), foreign);
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

    mlir::StringAttr OwnershipSignaturePass::searchedSymbol(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        auto searchOp = castOp ? castOp.getIn().getDefiningOp<mlir_ts::SearchForAddressOfSymbolOp>()
                               : mlir_ts::SearchForAddressOfSymbolOp();
        auto nameOp = searchOp ? searchOp->getOperand(0).getDefiningOp<mlir_ts::ConstantOp>() : mlir_ts::ConstantOp();
        auto name = nameOp ? mlir::dyn_cast<mlir::StringAttr>(nameOp.getValue()) : mlir::StringAttr();
        if (!name)
        {
            return {};
        }

        // only a function this module declares: its type is the one the call is made with
        auto funcOp = functions.lookup(name.getValue());
        return funcOp ? funcOp.getSymNameAttr() : mlir::StringAttr();
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
            // a class's `..instanceOf`, called through its vtable (`___cast` tests an instance so):
            // the compiler's own in every class, which destroys nothing, as the `.instanceOf` slot
            // read below - whoever overrides it, an exported class's too (TSLANG 45)
            if (identifier.getValue().ends_with("..instanceOf"))
            {
                callees.known = true;
                callees.instanceOf = true;
                return;
            }
            // a @final receiver's slot holds the method named: no family, one function
            if (!refOp->hasAttr(FINAL_RECEIVER_ATTR_NAME))
            {
                index = refOp.getIndex();
            }
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(def))
        {
            // an accessor's, read the same way
            identifier = refOp.getIdentifierAttr().getAttr();
            if (!refOp->hasAttr(FINAL_RECEIVER_ATTR_NAME))
            {
                index = refOp.getIndex();
            }
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::SymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (auto symbol = searchedSymbol(def))
        {
            // `ts.Cast(ts.SearchForAddressOfSymbol("C.set"))`: the indexer of a class imported from
            // a library, whose accessors are looked up by name (issue 43); that function, as a call
            // of `C.set` through its object
            identifier = symbol;
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
        else if (auto symbols = capturedFunctions(def); !symbols.empty())
        {
            // a function value a closure captured: `const f = (y) => ...; const g = (y) => f(y)`
            // (TSLANG 82), a call of the functions stored into its box's field
            callees.known = true;
            for (auto symbol : symbols)
            {
                callees.named.push_back(symbol);
                callees.known = (addDefined(symbol, callees) || addImported(symbol, callees)) && callees.known;
            }

            return;
        }
        else if (auto symbol = globalFunction(def))
        {
            // a global nothing assigns, given a function by its initializer: `const f = Math.fround`
            // (TSLANG 75), a call of that function
            identifier = symbol;
        }
        else if (mlir::isa_and_nonnull<mlir_ts::LoadOp>(def) || (def && isCall(def)))
        {
            // a callback parameter, field or local; or what a call returned (spec 56)
            resolveCallback(callee, callees);
            return;
        }
        else if (auto valueOp = mlir::dyn_cast_or_null<mlir_ts::ValueOp>(def);
                 valueOp && valueOp.getIn().getDefiningOp<mlir_ts::LoadOp>())
        {
            // an optional callback, called once tested
            resolveCallback(valueOp.getIn(), callees);
            return;
        }
        else if (auto boundOp = mlir::dyn_cast_or_null<mlir_ts::CreateBoundFunctionOp>(def))
        {
            // a function-typed field of an interface, called: `ts.CreateBoundFunction(
            // ts.ExtractInterfaceThis(i), ts.Load(ts.InterfaceSymbolRef(i, f)))`
            // (an optional method read out once checked, `ts.Value(ts.Load(...))`, TSLANG 187)
            auto func = boundOp.getFunc();
            if (auto valueOp = func.getDefiningOp<mlir_ts::ValueOp>())
            {
                func = valueOp.getIn();
            }

            auto loadOp = func.getDefiningOp<mlir_ts::LoadOp>();
            auto refOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>() : mlir_ts::InterfaceSymbolRefOp();
            auto thisOp = boundOp.getThisVal().getDefiningOp<mlir_ts::ExtractInterfaceThisOp>();
            if (refOp && isInterfaceField(refOp.getResult()) && thisOp && thisOp.getInterfaceVal() == refOp.getInterfaceVal())
            {
                resolveInterfaceSlot(refOp, interfaceFieldFamilies, callees);
                // ...or a slot every vtable fills with a function, not with a field of its object: the
                // thunk a generator is given a `Generator<T>` with (its `next(value?)` over the
                // generator's own `next()`)
                if (!callees.known && callees.named.empty())
                {
                    resolveInterfaceSlot(refOp, interfaceFamilies, callees);
                }
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

    llvm::SmallVector<mlir::StringAttr> OwnershipSignaturePass::capturedFunctions(mlir::Operation *def)
    {
        llvm::SmallVector<mlir::StringAttr> symbols;
        // `ts.Load(ts.PropertyRef(%box, n))`, the box the closure's first argument
        auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
        auto field = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto box = field ? mlir::dyn_cast<mlir::BlockArgument>(field.getObjectRef()) : mlir::BlockArgument();
        auto closure = box ? mlir::dyn_cast_or_null<mlir_ts::FuncOp>(box.getOwner()->getParentOp()) : mlir_ts::FuncOp();
        if (!closure || box.getArgNumber() != 0 || !box.getOwner()->isEntryBlock() || !mlir::isa<mlir_ts::RefType>(box.getType()))
        {
            return symbols;
        }

        auto position = field.getPosition();
        auto name = closure.getSymName();
        auto module = closure->getParentOfType<mlir::ModuleOp>();
        auto failed = false;
        llvm::SetVector<mlir::Value> boxes;
        llvm::SetVector<mlir::StringRef> closures;
        // the boxes the closure is made with, or called with
        module.walk([&](mlir::Operation *op) {
            if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(op))
            {
                auto symbolRefOp = boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
                if (symbolRefOp && symbolRefOp.getIdentifier() == name)
                {
                    boxes.insert(boundOp.getThisVal());
                }
            }
            else if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(op);
                     callOp && callOp.getCallee() == name && callOp->getNumOperands() > 0)
            {
                boxes.insert(callOp->getOperand(0));
            }
            else if (auto refOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(op); refOp && refOp.getIdentifier() == name)
            {
                // the closure's symbol used other than to bind it to a box: not known
                failed = failed || llvm::any_of(refOp->getUsers(), [](mlir::Operation *user) {
                             return !mlir::isa<mlir_ts::CreateBoundFunctionOp>(user);
                         });
            }
        });

        if (failed || boxes.empty())
        {
            return symbols;
        }

        // each box: a local only bound, called with and read for its fields, whose field `position` is
        // given a function's symbol and nothing else; the closures that get it write none of its fields
        for (auto boxValue : boxes)
        {
            auto varOp = boxValue.getDefiningOp<mlir_ts::VariableOp>();
            if (!varOp)
            {
                return {};
            }

            for (auto *user : varOp->getUsers())
            {
                if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user))
                {
                    auto symbolRefOp = boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
                    if (!symbolRefOp)
                    {
                        return {};
                    }

                    closures.insert(symbolRefOp.getIdentifier());
                    continue;
                }

                if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(user))
                {
                    if (llvm::count(callOp->getOperands(), varOp.getResult()) != 1 || callOp->getOperand(0) != varOp.getResult())
                    {
                        return {};
                    }

                    closures.insert(callOp.getCallee());
                    continue;
                }

                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    return {};
                }

                for (auto *refUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(refUser);
                    if (mlir::isa<mlir_ts::LoadOp>(refUser) ||
                        (storeOp && propertyRefOp.getPosition() != position && storeOp.getReference() == propertyRefOp.getResult()))
                    {
                        continue;
                    }

                    if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                    {
                        return {};
                    }

                    auto stored = storeOp.getValue();
                    while (auto castOp = stored.getDefiningOp<mlir_ts::CastOp>())
                    {
                        stored = castOp.getIn();
                    }

                    auto symbolRefOp = stored.getDefiningOp<mlir_ts::SymbolRefOp>();
                    if (!symbolRefOp)
                    {
                        return {};
                    }

                    if (!llvm::is_contained(symbols, symbolRefOp.getIdentifierAttr().getAttr()))
                    {
                        symbols.push_back(symbolRefOp.getIdentifierAttr().getAttr());
                    }
                }
            }
        }

        // the closures that get the box write none of its fields
        for (auto closureName : closures)
        {
            auto funcOp = functions.lookup(closureName);
            if (!funcOp || funcOp.isExternal())
            {
                return {};
            }

            auto &entry = funcOp.getFunctionBody().front();
            if (entry.getNumArguments() == 0)
            {
                return {};
            }

            auto written = false;
            funcOp.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
                written = written || (propertyRefOp.getObjectRef() == entry.getArgument(0) &&
                                      llvm::any_of(propertyRefOp->getUsers(),
                                                   [](mlir::Operation *user) { return !mlir::isa<mlir_ts::LoadOp>(user); }));
            });
            if (written)
            {
                return {};
            }
        }

        return symbols;
    }

    mlir::StringAttr OwnershipSignaturePass::globalFunction(mlir::Operation *def)
    {
        // `ts.Load(ts.AddressOf @g)` of a global no store writes, whose initializer gives it a
        // function's symbol, seen through its casts
        auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
        auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
        if (!addressOfOp || writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            return {};
        }

        mlir::Value value;
        // ...or a module `const` beside top-level code, which the root's code gives its value (TSLANG 191)
        if (auto stored = writtenOnceStores.find(addressOfOp.getGlobalName()); stored != writtenOnceStores.end())
        {
            value = stored->second.getValue();
        }
        else
        {
            auto globalOp = globals.lookup(addressOfOp.getGlobalName());
            auto *block = globalOp ? globalOp.getInitializerBlock() : nullptr;
            auto resultOp = block && !block->empty() ? mlir::dyn_cast<mlir_ts::GlobalResultOp>(block->getTerminator()) : mlir_ts::GlobalResultOp();
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return {};
            }

            value = resultOp->getOperand(0);
        }

        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            value = castOp.getIn();
        }

        auto symbolRefOp = value.getDefiningOp<mlir_ts::SymbolRefOp>();
        return symbolRefOp ? symbolRefOp.getIdentifierAttr().getAttr() : mlir::StringAttr();
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
