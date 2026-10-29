// Class declaration/member/vtable code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    mlir::LogicalResult MLIRGenImpl::registerGenericClass(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = className(classDeclarationAST, genContext);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericClassesMap.count(fullNamePtr))
            {
                // already registered - but the registration itself typically happens during
                // Stages::Discovering (before addGenericClassDeclarationToExport's
                // isAddedToExport gate, which only actually emits once stage ==
                // Stages::SourceGeneration, will do anything) - retry the export step alone
                // using the existing GenericClassInfo rather than skipping it entirely.
                if (getExportModifier(classDeclarationAST))
                {
                    addGenericClassDeclarationToExport(fullNameGenericClassesMap.lookup(fullNamePtr));
                }

                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(processTypeParameters(classDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            // register class
            GenericClassInfo::TypePtr newGenericClassPtr = std::make_shared<GenericClassInfo>();
            newGenericClassPtr->name = namePtr;
            newGenericClassPtr->fullName = fullNamePtr;
            newGenericClassPtr->typeParams = typeParameters;
            newGenericClassPtr->classDeclaration = classDeclarationAST;
            newGenericClassPtr->elementNamespace = currentNamespace;
            newGenericClassPtr->sourceFile = sourceFile;
            newGenericClassPtr->fileName = mainSourceFileName;

            mlirGenClassType(newGenericClassPtr, genContext);

            getGenericClassesMap().insert({namePtr, newGenericClassPtr});
            fullNameGenericClassesMap.insert(fullNamePtr, newGenericClassPtr);

            // support dynamic loading: a generic class is never instantiated in this module
            // if nothing here uses it concretely, so mlirGen(ClassLikeDeclaration) never
            // reaches its own addClassDeclarationToExport call below (that only runs for a
            // SPECIALIZED instantiation, gated on genContext.typeParamsWithArgs being
            // non-empty) - the bare template needs to be exported here instead, the one
            // place every generic declaration passes through regardless of whether it is
            // ever instantiated locally.
            if (getExportModifier(classDeclarationAST))
            {
                addGenericClassDeclarationToExport(newGenericClassPtr);
            }

            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ClassDeclaration classDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(theModule.getBody());

        auto value = mlirGen(classDeclarationAST.as<ClassLikeDeclaration>(), genContext);
        return std::get<0>(value);
    }

    std::pair<mlir::LogicalResult, mlir::StringRef> MLIRGenImpl::mlirGen(ClassLikeDeclaration classDeclarationAST,
                                                            const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        auto isGenericClass = classDeclarationAST->typeParameters.size() > 0;
        if (isGenericClass && genContext.typeParamsWithArgs.size() == 0)
        {
            // a value is one layout: each instantiation would be another type to copy
            auto isValueClass = false;
            iterateDecorators(classDeclarationAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                isValueClass = isValueClass || name == VALUE_DECORATOR;
            });
            if (isValueClass)
            {
                emitError(loc(classDeclarationAST)) << "@" VALUE_DECORATOR " class '"
                                                    << MLIRHelper::getName(classDeclarationAST->name)
                                                    << "' cannot have type parameters";
                return {mlir::failure(), ""};
            }

            return {registerGenericClass(classDeclarationAST, genContext), ""};
        }

        auto newClassPtr = mlirGenClassInfo(classDeclarationAST, genContext);
        if (!newClassPtr)
        {
            return {mlir::failure(), ""};
        }

        // an abstract class is there to be extended, and a C++ binding is called through the C++
        // vtable: neither can be @final
        if (newClassPtr->isFinal && (newClassPtr->isAbstract || newClassPtr->isCppVTable))
        {
            emitError(loc(classDeclarationAST))
                << "class '" << newClassPtr->name << "' cannot be both @" FINAL_DECORATOR " and "
                << (newClassPtr->isAbstract ? "abstract" : "@" CPP_VTABLE);
            return {mlir::failure(), ""};
        }

        // a value has no vtable, so it cannot be abstract, a C++ binding or @final (nothing extends
        // a value: saying so twice would suggest it could be)
        if (newClassPtr->isValue && (newClassPtr->isAbstract || newClassPtr->isCppVTable || newClassPtr->isFinal))
        {
            emitError(loc(classDeclarationAST))
                << "class '" << newClassPtr->name << "' cannot be both @" VALUE_DECORATOR " and "
                << (newClassPtr->isAbstract ? "abstract" : newClassPtr->isCppVTable ? "@" CPP_VTABLE : "@" FINAL_DECORATOR);
            return {mlir::failure(), ""};
        }

        if (isGenericClass && genContext.typeParamsWithArgs.size() > 0)
        {
            // a concrete instantiation of a generic class (e.g. Box<number>) reprocesses the
            // SAME class declaration AST as the bare template (e.g. `export class Box<T>`),
            // so mlirGenClassInfo's isExport = getExportModifier(classDeclarationAST) above
            // would otherwise mark this LOCAL, per-instantiation specialization as exported
            // too - wrong on two counts: (1) semantically, a specialization materialized by
            // whichever module instantiates it is local to that module, not a re-export of
            // it (an importer using M.Box<number> isn't re-exporting M.Box<number> further);
            // (2) mechanically, a multi-type-param instantiation's mangled name contains a
            // raw comma (e.g. M.Pair<!ts.number,!ts.string>..instanceOf), which is a
            // metacharacter in the linker's `/EXPORT:name[,option]` directive syntax - lld
            // rejects it outright ("invalid /export:") the moment anything in it is
            // actually marked for export. The generic TEMPLATE's own declaration is already
            // exported correctly and separately (see registerGenericClass /
            // addGenericClassDeclarationToExport), which is the only export this class
            // needs.
            newClassPtr->isExport = false;
            newClassPtr->isPublic = false;

            // do not process specialized class second time;
            // TODO: investigate why classType is provided already for class
            // Not across modules, though: an import met during discovery is generated in full, but
            // into the discovery module, which is erased. The members of a specialization made there
            // are gone, and the module that uses it has to have its own.
            if (testProcessingState(newClassPtr, ProcessingStages::Processing, genContext)
                && (genContext.allowPartialResolve || newClassPtr->processingModule == theModule.getOperation()))
            {
                return {mlir::success(), newClassPtr->classType.getName().getValue()};
            }
        }

        // One library of several modules can declare an imported class more than once: a module
        // whose export depends on a class of another module declares that class too (a derived
        // class's base), so the importer reads `Base` from both modules' __decls. The second is
        // the same external class again, and generating it again redefined `Base..new`.
        if (!isGenericClass && newClassPtr->isImport
            && newClassPtr->processingDeclaration && newClassPtr->processingDeclaration != classDeclarationAST
            && newClassPtr->processingModule == theModule.getOperation()
            && testProcessingState(newClassPtr, ProcessingStages::Processed, genContext))
        {
            return {mlir::success(), newClassPtr->classType.getName().getValue()};
        }

        if (!genContext.allowPartialResolve)
        {
            newClassPtr->processingModule = theModule.getOperation();
            newClassPtr->processingDeclaration = classDeclarationAST;
        }

        setProcessingState(newClassPtr, ProcessingStages::Processing, genContext);

        auto location = loc(classDeclarationAST);

        if (mlir::succeeded(mlirGenClassType(newClassPtr, genContext)))
        {
            newClassPtr->typeParamsWithArgs = genContext.typeParamsWithArgs;
            // here, before anything of the type is generated: whether a reference owns anything
            // (MLIRTypeHelper::ownsHeapMemory) is read off the type, and a retain placed while it
            // still said yes would never pair with a release
            mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType()).setCpp(newClassPtr->isCppVTable);
        }

        // if this is generic specialized class then do not generate code for it
        if (mth.isGenericType(newClassPtr->classType))
        {
            return {mlir::success(), newClassPtr->classType.getName().getValue()};
        }

        // init this type (needed to use in property evaluations)
        GenContext classGenContext(genContext);
        classGenContext.thisType = newClassPtr->classType;
        classGenContext.thisClassType = newClassPtr->classType;
        classGenContext.specialization = isGenericClass;

        // we need THIS in params
        SymbolTableScopeT varScope(symbolTable);
        resetScope();   

        setProcessingState(newClassPtr, ProcessingStages::ProcessingStorageClass, genContext);
        newClassPtr->fieldsTypedAfterMembers.clear();
        if (mlir::failed(mlirGenClassStorageType(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInStorageClass, genContext);
            return {mlir::failure(), ""};
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessedStorageClass, genContext);

        // the base classes are known now, and no method has been named yet
        if (newClassPtr->isCppVTable)
        {
            mlirGenClassCppOverloadNames(classDeclarationAST, newClassPtr, classGenContext);
        }

        mlirGenClassConstructorOverloadNames(classDeclarationAST);

        if (mlir::failed(mlirGenClassCppVTableCheck(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInStorageClass, genContext);
            return {mlir::failure(), ""};
        }

        // if it is ClassExpression we need to know if it is declaration
        mlirGenClassCheckIfDeclaration(location, classDeclarationAST, newClassPtr, classGenContext);

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        llvm::SmallVector<bool> membersProcessStates;
        if (isGenericClass)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(theModule.getBody());

            saveMembersProcessStates(classDeclarationAST, newClassPtr, membersProcessStates);

            // before processing generic class for example array<int> array<string> we need to drop all states of processed members
            clearMembersProcessStates(classDeclarationAST, newClassPtr);
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessingBody, genContext);

        // prepare VTable
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo, 1> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        // a @cpp class is never constructed, typed or allocated here: its objects come from C++;
        // unless it implements C++ methods, which it does in an object of its own
        auto hasOwnRuntimeMembers = !newClassPtr->isStatic && (!newClassPtr->isCppVTable || newClassPtr->isCppVTableImplementation);

        if (hasOwnRuntimeMembers)
        {
            mlirGenClassDefaultConstructor(classDeclarationAST, newClassPtr, classGenContext);
        }

        // a @value class has a constructor and nothing an object has: no vtable, no `.instanceOf`,
        // no `..new` - `new` makes the value in place (NewValueClassInstance)
        auto hasObjectMembers = hasOwnRuntimeMembers && !newClassPtr->isValue;

#ifdef ENABLE_RTTI
        // no `.instanceOf`: a C++ vtable has only the C++ methods
        if (hasObjectMembers && !newClassPtr->isCppVTable)
        {
            // INFO: .instanceOf must be first element in VTable for Cast Any
            mlirGenClassInstanceOfMethod(classDeclarationAST, newClassPtr, classGenContext);
        }
#endif

#if ENABLE_TYPED_GC
        auto enabledGC = compileOptions.needsGCRuntime();
        if (enabledGC && hasObjectMembers)
        {
            mlirGenClassTypeBitmap(location, newClassPtr, classGenContext);
            mlirGenClassTypeDescriptorField(location, newClassPtr, classGenContext);
        }
#endif

        if (hasObjectMembers)
        {
            mlirGenClassNew(classDeclarationAST, newClassPtr, classGenContext);
        }

        mlirGenClassDefaultStaticConstructor(classDeclarationAST, newClassPtr, classGenContext);

        /*
        // to support call 'static v = new Class();'
        if (mlir::failed(mlirGenClassStaticFields(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            return {mlir::failure(), ""};
        }
        */

        if (mlir::failed(mlirGenClassMembers(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInMembers, genContext);
            return {mlir::failure(), ""};
        }

        if (mlir::failed(mlirGenClassRetypeFields(newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInMembers, genContext);
            return {mlir::failure(), ""};
        }

        // the slots of a @cpp class's methods, each where its @slot(N) puts it: two in one slot is
        // an error (getVirtualTable finds it, here it's reported)
        if (newClassPtr->isCppVTable)
        {
            llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo, 1> cppVirtualTable;
            newClassPtr->cppSlotConflict.clear();
            newClassPtr->getVirtualTable(cppVirtualTable);
            if (!newClassPtr->cppSlotConflict.empty())
            {
                emitError(location) << "@" CPP_SLOT ": " << newClassPtr->cppSlotConflict;
                setProcessingState(newClassPtr, ProcessingStages::ErrorInMembers, genContext);
                return {mlir::failure(), ""};
            }
        }

        // generate vtable for interfaces in base class
        if (mlir::failed(mlirGenClassBaseInterfaces(location, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInBaseInterfaces, genContext);
            return {mlir::failure(), ""};
        }

        // generate vtable for interfaces
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClauseImplements(classDeclarationAST, newClassPtr, heritageClause,
                                                                  classGenContext)))
            {
                setProcessingState(newClassPtr, ProcessingStages::ErrorInHeritageClauseImplements, genContext);
                return {mlir::failure(), ""};
            }
        }

        if (hasObjectMembers)
        {
            if (mlir::failed(mlirGenClassVirtualTableDefinition(location, newClassPtr, classGenContext)))
            {
                setProcessingState(newClassPtr, ProcessingStages::ErrorInVTable, genContext);
                return {mlir::failure(), ""};
            }
        }

        // here we need to process New method;

        if (isGenericClass)
        {
            builder.restoreInsertionPoint(savePoint);

            restoreMembersProcessStates(classDeclarationAST, newClassPtr, membersProcessStates);
            //LLVM_DEBUG(llvm::dbgs() << "\n>>>>>>>>>>>>>>>>> module: \n" << theModule << "\n";);
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessedBody, genContext);

        // support dynamic loading
        if (getExportModifier(classDeclarationAST))
        {
            addClassDeclarationToExport(newClassPtr);
        }

        setProcessingState(newClassPtr, ProcessingStages::Processed, genContext);

        return {mlir::success(), newClassPtr->classType.getName().getValue()};
    }

    ClassInfo::TypePtr MLIRGenImpl::mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        return mlirGenClassInfo(className(classDeclarationAST, genContext), classDeclarationAST, genContext);
    }

    ClassInfo::TypePtr MLIRGenImpl::mlirGenClassInfo(const std::string &name, ClassLikeDeclaration classDeclarationAST,
                                        const GenContext &genContext)
    {
        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        ClassInfo::TypePtr newClassPtr;
        if (fullNameClassesMap.count(fullNamePtr))
        {
            newClassPtr = fullNameClassesMap.lookup(fullNamePtr);
            getClassesMap().insert({namePtr, newClassPtr});
        }
        else
        {
            // register class
            newClassPtr = std::make_shared<ClassInfo>();
            newClassPtr->name = namePtr;
            newClassPtr->fullName = fullNamePtr;
            newClassPtr->elementNamespace = currentNamespace;
            newClassPtr->isAbstract = hasModifier(classDeclarationAST, SyntaxKind::AbstractKeyword);
            newClassPtr->isDeclaration =
                declarationMode || hasModifier(classDeclarationAST, SyntaxKind::DeclareKeyword);
            newClassPtr->isStatic = hasModifier(classDeclarationAST, SyntaxKind::StaticKeyword);
            newClassPtr->isExport = getExportModifier(classDeclarationAST);
            newClassPtr->isPublic = hasModifier(classDeclarationAST, SyntaxKind::ExportKeyword);
            newClassPtr->hasVirtualTable = newClassPtr->isAbstract;

            // check decorator for class
            iterateDecorators(classDeclarationAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                if (name == DLL_EXPORT)
                {
                    newClassPtr->isExport = true;
                }

                if (name == DLL_IMPORT)
                {
                    newClassPtr->isDeclaration = true;
                    newClassPtr->isImport = true;
                    // it has parameter, means this is dynamic import, should point to dll path
                    if (args.size() > 0)
                    {
                        newClassPtr->isDynamicImport = true;
                    }
                }

                // a binding to C++ objects: declared (its methods have no bodies), with a vtable
                // field even with no `abstract` method (see ClassInfo::isCppVTable); or, with
                // methods that have bodies, C++ methods implemented here
                // (ClassInfo::isCppVTableImplementation)
                if (name == CPP_VTABLE)
                {
                    newClassPtr->isCppVTable = true;
                    newClassPtr->hasVirtualTable = true;
                    newClassPtr->isCppVTableImplementation = llvm::any_of(classDeclarationAST->members, [](ClassElement member) {
                        return member == SyntaxKind::MethodDeclaration && member.as<MethodDeclaration>()->body;
                    });
                    if (!newClassPtr->isCppVTableImplementation)
                    {
                        newClassPtr->isDeclaration = true;
                    }
                }

                if (name == FINAL_DECORATOR)
                {
                    newClassPtr->isFinal = true;
                }

                if (name == VALUE_DECORATOR)
                {
                    newClassPtr->isValue = true;
                }
            });

            getClassesMap().insert({namePtr, newClassPtr});
            fullNameClassesMap.insert(fullNamePtr, newClassPtr);
        }

        return newClassPtr;
    }

    // C++ overloads: a @cpp class may declare one name several times, each a method of its own (its
    // own @slot or @linkname). Methods are known by name, so an overload is renamed `name#1`,
    // `name#2`, ... - `#` is in no identifier - and a call of `name` picks one by its arguments
    // (resolveCppOverload). DeclarationPrinter prints the name before the `#`.
    //
    // A method in the slot of a base class's overload takes that overload's name, which is how it
    // overrides it; any other method whose name is taken - by an earlier declaration here, or, with
    // a @slot of its own, by a base class's method - takes the next free `name#N`. Run again on the
    // same declaration (discovery, an import), it leaves the names as they are.
    void MLIRGenImpl::mlirGenClassCppOverloadNames(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                   const GenContext &genContext)
    {
        struct BaseMethod
        {
            std::string name;
            int slot;
        };

        llvm::StringMap<SmallVector<BaseMethod>> baseMethods;
        std::function<void(ClassInfo::TypePtr)> collect = [&](ClassInfo::TypePtr classInfo) {
            for (auto &base : classInfo->baseClasses)
            {
                for (auto &method : base->methods)
                {
                    if (!method.isStatic)
                    {
                        baseMethods[cppOverloadedName(method.name)].push_back({method.name, method.cppSlot});
                    }
                }

                collect(base);
            }
        };
        collect(newClassPtr);

        auto isBaseName = [&](StringRef name) {
            auto it = baseMethods.find(cppOverloadedName(name));
            return it != baseMethods.end()
                && llvm::any_of(it->second, [&](BaseMethod &method) { return method.name == name; });
        };

        SmallVector<MethodDeclaration> methods;
        llvm::StringSet<> ownNames;
        for (auto &classMember : classDeclarationAST->members)
        {
            if (classMember == SyntaxKind::MethodDeclaration && !hasModifier(classMember, SyntaxKind::StaticKeyword)
                && classMember.as<MethodDeclaration>()->name == SyntaxKind::Identifier)
            {
                auto methodDeclaration = classMember.as<MethodDeclaration>();
                methods.push_back(methodDeclaration);
                auto name = MLIRHelper::getName(methodDeclaration->name);
                if (name.find(CPP_OVERLOAD_SEPARATOR) != std::string::npos)
                {
                    ownNames.insert(name);
                }
            }
        }

        NodeFactory nf(NodeFactoryFlags::None);
        auto rename = [&](MethodDeclaration methodDeclaration, StringRef newName) {
            auto renamed = nf.createIdentifier(stows(newName.str()));
            renamed->pos = methodDeclaration->name->pos;
            renamed->_end = methodDeclaration->name->_end;
            methodDeclaration->name = renamed;
        };

        for (auto methodDeclaration : methods)
        {
            auto name = MLIRHelper::getName(methodDeclaration->name);
            if (name.find(CPP_OVERLOAD_SEPARATOR) != std::string::npos)
            {
                continue;
            }

            auto slot = getCppSlot(methodDeclaration, genContext);
            if (slot >= 0)
            {
                // overriding a base class's overload: its name
                auto it = baseMethods.find(name);
                if (it != baseMethods.end())
                {
                    auto overridden = llvm::find_if(it->second, [&](BaseMethod &method) { return method.slot == slot; });
                    if (overridden != it->second.end())
                    {
                        if (overridden->name != name)
                        {
                            rename(methodDeclaration, overridden->name);
                        }

                        ownNames.insert(overridden->name);
                        continue;
                    }
                }
            }

            // a method without @slot keeps a base class's name: it overrides it by name, or is
            // reported (mlirGenClassCppVTableCheck)
            auto taken = ownNames.contains(name) || (slot >= 0 && isBaseName(name));
            if (!taken)
            {
                ownNames.insert(name);
                continue;
            }

            for (auto index = 1;; index++)
            {
                auto candidate = name + CPP_OVERLOAD_SEPARATOR + std::to_string(index);
                if (!ownNames.contains(candidate) && !isBaseName(candidate))
                {
                    rename(methodDeclaration, candidate);
                    ownNames.insert(candidate);
                    break;
                }
            }
        }
    }

    // Constructor overloads: a class may implement several constructors, each a function of its own.
    // The first is `constructor`, as with one, so whatever looks the constructor up by name (super
    // calls, a default constructor, a cast) finds it; any other is renamed `constructor#1`,
    // `constructor#2`, ... as C++ overloads are, and `new` picks one by its arguments
    // (resolveConstructorOverload). Run again on the same declaration (a specialization of a generic
    // class, an import), it leaves the names as they are.
    void MLIRGenImpl::mlirGenClassConstructorOverloadNames(ClassLikeDeclaration classDeclarationAST)
    {
        NodeFactory nf(NodeFactoryFlags::None);
        auto index = 0;
        for (auto &classMember : classDeclarationAST->members)
        {
            // an overload signature (no body) is a type only, not a constructor of its own
            if (classMember != SyntaxKind::Constructor || hasModifier(classMember, SyntaxKind::StaticKeyword)
                || !classMember.as<ConstructorDeclaration>()->body)
            {
                continue;
            }

            auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
            if (index++ == 0 || constructorDeclaration->name)
            {
                continue;
            }

            auto renamed = nf.createIdentifier(stows(std::string(CONSTRUCTOR_NAME CPP_OVERLOAD_SEPARATOR) + std::to_string(index - 1)));
            renamed->pos = constructorDeclaration->pos;
            renamed->_end = constructorDeclaration->_end;
            constructorDeclaration->name = renamed;
        }
    }

    // The name a C++ overload was declared with: `write#1` is `write`
    StringRef MLIRGenImpl::cppOverloadedName(StringRef methodName)
    {
        return methodName.split(CPP_OVERLOAD_SEPARATOR).first;
    }

    // MSVC returns a C++ record (struct, class - RefCountPtr<T> too) from an instance method through
    // a hidden pointer, passed after `this` and returned in turn (clang's
    // MicrosoftCXXABI::classifyReturnType: "MSVC always returns structs indirectly from C++ instance
    // methods"). A @cpp method declared to return a tuple returns that record: its slot is
    // called as (this, Ref<T>, args...) -> Ref<T>, the type kept in the MethodInfo, and a call
    // (mlirGenCall) passes a local of T and reads it back. The declared type stays for
    // DeclarationPrinter, so an importer lowers it again.
    //
    // The Itanium ABI on x86-64 and AArch64 (clang's ItaniumCXXABI::classifyReturnType, then the
    // SysV or AAPCS64 classification) passes the pointer before `this`, for a record with a
    // destructor or a copy constructor, or of more than 16 bytes - the method's @cppnontrivial, as a
    // tuple can't say so - and the call swaps the two and marks the pointer `sret` (X8 on AArch64).
    // A trivial record of up to 16 bytes of integers and pointers comes back in RAX:RDX / X0:X1, the
    // method keeping its declared type: the call reads the registers back as the record
    // (LowerToLLVM's lowerCppCallArgs). One with floating-point fields isn't done here.
    //
    // A record passed by value (a tuple parameter) is passed as each ABI does, by the call's lowering
    // too (lowerCppCallArgs).
    //
    // A method with a body (of a class implementing C++ methods) is called by C++: it is marked
    // CPP_METHOD_ATTR_NAME, and can't take or return a record yet (its body would have to read it
    // as the ABI passes it, write it through the hidden pointer); nor can a C++ non-virtual method,
    // called by its symbol.
    mlir::LogicalResult MLIRGenImpl::mlirGenClassCppVTableHiddenReturn(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                          mlir_ts::FuncOp funcOp, bool nonTrivial, bool hasBody,
                                                          bool isVirtual)
    {
        auto funcName = funcOp.getName();
        // a method with a body that isn't virtual is TypeScript's, not a C++ method
        if (!newClassPtr->isCppVTable || (hasBody && !isVirtual))
        {
            if (nonTrivial)
            {
                emitError(location) << "@" CPP_NON_TRIVIAL " is for a method of a @" CPP_VTABLE " class returning a struct";
                return mlir::failure();
            }

            return mlir::success();
        }

        cppVTableMethods.insert(funcName);
        if (hasBody)
        {
            funcOp->setAttr(CPP_METHOD_ATTR_NAME, builder.getUnitAttr());
        }
        else
        {
            cppMethodsWithoutBody.insert(funcName);
        }

        auto method = llvm::find_if(newClassPtr->methods, [&](auto &method) { return method.funcName == funcName; });
        if (method == newClassPtr->methods.end() || method->declaredFuncType)
        {
            return mlir::success();
        }

        auto funcType = method->funcType;
        if ((hasBody || !isVirtual)
            && llvm::any_of(funcType.getInputs().drop_front(), [](mlir::Type type) { return isa<mlir_ts::TupleType>(type); }))
        {
            emitError(location) << (hasBody ? "a method implementing a C++ method (a @" CPP_VTABLE " class's, with a body)"
                                            : "a C++ non-virtual method (a @" CPP_VTABLE " class's, without @" CPP_SLOT ")")
                                << " can't take a struct by value yet";
            return mlir::failure();
        }

        if (funcType.getNumResults() != 1 || !isa<mlir_ts::TupleType>(funcType.getResult(0)) || funcType.getNumInputs() < 1)
        {
            if (nonTrivial)
            {
                emitError(location) << "@" CPP_NON_TRIVIAL " is for a method of a @" CPP_VTABLE " class returning a struct";
                return mlir::failure();
            }

            return mlir::success();
        }

        if (hasBody)
        {
            emitError(location) << "a method implementing a C++ method (a @" CPP_VTABLE " class's, with a body) can't return a struct yet";
            return mlir::failure();
        }

        if (!isVirtual)
        {
            emitError(location) << "a C++ non-virtual method (a @" CPP_VTABLE " class's, without @" CPP_SLOT ") can't return a struct yet";
            return mlir::failure();
        }

        auto resultType = funcType.getResult(0);
        auto &targetInfo = compileOptions.targetInfo;
        auto beforeThis = false;
        method->cppNonTrivialReturn = nonTrivial;
        if (targetInfo.hiddenReturnPointerBeforeThis)
        {
            if (!nonTrivial)
            {
                if (isCppRecordReturnedInIntegerRegisters(mlir::cast<mlir_ts::TupleType>(resultType)))
                {
                    return mlir::success();
                }

                emitError(location) << "on this target a method of a @" CPP_VTABLE " class returns a struct in registers by its fields, "
                                    << "which is supported for up to 16 bytes of integers and pointers, not floating-point ones; one "
                                    << "with a destructor or a copy constructor, or of more than 16 bytes, is returned through a "
                                    << "pointer: mark the method @" CPP_NON_TRIVIAL;
                return mlir::failure();
            }

            beforeThis = true;
        }
        else if (!targetInfo.hiddenReturnPointerAfterThis)
        {
            emitError(location) << "a method of a @" CPP_VTABLE " class returning a struct needs the MSVC ABI, or the Itanium ABI on "
                                << "x86-64 or AArch64, which return it through a pointer next to 'this'";
            return mlir::failure();
        }

        auto resultRefType = mlir_ts::RefType::get(resultType);
        SmallVector<mlir::Type> inputs;
        inputs.push_back(funcType.getInput(0));
        inputs.push_back(resultRefType);
        inputs.append(funcType.getInputs().begin() + 1, funcType.getInputs().end());

        method->declaredFuncType = funcType;
        method->funcType = getFunctionType(inputs, {resultRefType}, funcType.isVarArg());
        cppVTableHiddenReturns[funcName] = resultType;
        if (beforeThis)
        {
            cppVTableHiddenReturnsBeforeThis.insert(funcName);
        }

        return mlir::success();
    }

    // The SysV (AAPCS64) classification of a trivial C++ record, as far as tslang returns a tuple the
    // same way: one or two eightbytes of class INTEGER, each a whole field (an LLVM struct of i64s
    // and pointers comes back in RAX:RDX / X0:X1; smaller fields would be packed into one register).
    // Whether a record is made of integers and pointers only (booleans, a pointer that may be null,
    // records of them too), as the Itanium ABI returns in general purpose registers; its size, up
    // to 16 bytes, is the call lowering's to check (lowerCppCallArgs)
    bool MLIRGenImpl::isCppRecordReturnedInIntegerRegisters(mlir_ts::TupleType recordType)
    {
        auto fields = recordType.getFields();
        if (fields.empty())
        {
            return false;
        }

        std::function<bool(mlir::Type)> isInteger = [&](mlir::Type type) {
            if (isa<mlir::IntegerType, mlir_ts::BooleanType, mlir_ts::OpaqueType, mlir_ts::ClassType, mlir_ts::RefType,
                    mlir::IndexType>(type))
            {
                return true;
            }

            // Opaque | null: a pointer that may be null
            if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
            {
                return llvm::all_of(unionType.getTypes(), [&](mlir::Type member) {
                    return isa<mlir_ts::NullType>(member) || isInteger(member);
                });
            }

            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                return isCppRecordReturnedInIntegerRegisters(tupleType);
            }

            return false;
        };

        return llvm::all_of(fields, [&](auto &field) { return isInteger(field.type); });
    }

    // A @cpp class only names the slots of a vtable made by C++, so everything that would add
    // a slot, a field initializer or code of its own is an error, and so is a hierarchy mixing it
    // with ordinary classes: a slot number off by one calls the wrong C++ method, with no error.
    mlir::LogicalResult MLIRGenImpl::mlirGenClassCppVTableCheck(mlir::Location location,
                                                   ClassLikeDeclaration classDeclarationAST,
                                                   ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        if (MLIRHelper::hasDecorator(classDeclarationAST, CPP_VTABLE_OLD_NAME))
        {
            emitError(location) << "@" CPP_VTABLE_OLD_NAME " is now @" CPP_VTABLE ", and a virtual method needs @" CPP_SLOT "(N)";
            return mlir::failure();
        }

        if (!newClassPtr->isCppVTable)
        {
            // an ordinary class can't add to a C++ vtable
            for (auto &base : newClassPtr->baseClasses)
            {
                if (base->isCppVTable)
                {
                    emitError(location) << "class '" << newClassPtr->fullName << "' can't extend the @" CPP_VTABLE " class '"
                                        << base->fullName << "': it has to be @" CPP_VTABLE " too";
                    return mlir::failure();
                }
            }

            for (auto &classMember : classDeclarationAST->members)
            {
                if (MLIRHelper::hasDecorator(classMember, CPP_SLOT))
                {
                    emitError(loc(classMember)) << "@" CPP_SLOT " is for a method of a @" CPP_VTABLE " class";
                    return mlir::failure();
                }
            }

            return mlir::success();
        }

        auto failed = false;
        auto error = [&](mlir::Location errorLocation) -> mlir::InFlightDiagnostic {
            failed = true;
            return emitError(errorLocation);
        };

        if (compileOptions.targetInfo.memberCallsUseThiscall)
        {
            error(location) << "a @" CPP_VTABLE " class can't be used on this target: its C++ methods are __thiscall";
        }

        if (newClassPtr->isStatic)
        {
            error(location) << "a static class can't be @" CPP_VTABLE;
        }

        if (classDeclarationAST->typeParameters.size() > 0)
        {
            error(location) << "a generic class can't be @" CPP_VTABLE;
        }

        for (auto &base : newClassPtr->baseClasses)
        {
            if (!base->isCppVTable)
            {
                error(location) << "the @" CPP_VTABLE " class '" << newClassPtr->fullName << "' can only extend a @" CPP_VTABLE
                                << " class, not '" << base->fullName << "'";
            }
        }

        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (heritageClause->token == SyntaxKind::ImplementsKeyword)
            {
                error(loc(heritageClause)) << "a @" CPP_VTABLE " class can't implement an interface";
            }
        }

        // implementing C++ methods (isCppVTableImplementation): bodies, a constructor and field
        // initializers are its own; it is not exported, as an importer would get its declaration
        // only, with the methods' bodies but not the vtable they are in
        auto implementation = newClassPtr->isCppVTableImplementation;
        if (implementation && getExportModifier(classDeclarationAST))
        {
            error(location) << "a @" CPP_VTABLE " class implementing C++ methods can't be exported";
        }

        for (auto &classMember : classDeclarationAST->members)
        {
            auto memberLocation = loc(classMember);
            if (hasModifier(classMember, SyntaxKind::StaticKeyword))
            {
                error(memberLocation) << "a @" CPP_VTABLE " class can't have static members";
                continue;
            }

            // a method with @slot(N) is a C++ virtual method, as is one overriding a base class's
            // slot; with a body, it implements it (isCppVTableImplementation). Without @slot, a
            // method with a body is a TypeScript method of the class, and one without is a C++
            // non-virtual method, called by its symbol.
            if (classMember == SyntaxKind::MethodDeclaration)
            {
                auto slot = getCppSlot(classMember, genContext);
                if (slot == -2)
                {
                    error(memberLocation) << "@" CPP_SLOT " takes the slot number, an integer from 0";
                    continue;
                }

                auto methodName = getCppMethodName(classMember, genContext);
                auto overrides = isCppBaseSlot(newClassPtr, methodName);
                if (slot < 0 && overrides && isCppBaseSlot(newClassPtr, (cppOverloadedName(methodName) + CPP_OVERLOAD_SEPARATOR "1").str()))
                {
                    // by name, it would take the slot of the first of them
                    error(memberLocation) << "'" << cppOverloadedName(methodName) << "' is overloaded in a base class: an override "
                                          << "names the slot it overrides, @" CPP_SLOT "(N)";
                    continue;
                }
                if (slot < 0 && !overrides)
                {
                    auto hasBody = static_cast<bool>(classMember.as<MethodDeclaration>()->body);
                    if (hasModifier(classMember, SyntaxKind::AbstractKeyword))
                    {
                        error(memberLocation) << "the abstract method '" << cppOverloadedName(methodName) << "' of a @" CPP_VTABLE
                                              << " class needs @" CPP_SLOT "(N), its vtable slot";
                    }
                    else if (!hasBody && !newClassPtr->isImport && !declarationMode)
                    {
                        auto hasLinkName = false;
                        iterateDecorators(classMember, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                            // a @cppname that doesn't mangle is reported where the method is made
                            hasLinkName |= (name == DLL_NAME || name == CPP_NAME) && !args.empty();
                        });

                        if (!hasLinkName)
                        {
                            error(memberLocation) << "'" << cppOverloadedName(methodName) << "' has no @" CPP_SLOT ", so it isn't virtual: give a "
                                                  << "C++ virtual method its slot, @" CPP_SLOT "(N), or a non-virtual one its "
                                                  << "symbol, @" LINK_NAME "(\"...\"), or its C++ declaration, @" CPP_NAME "(\"...\")";
                        }
                    }
                }

                continue;
            }

            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                // a field of the C++ object, after its vtable pointer
                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (propertyDeclaration->initializer && !implementation)
                {
                    error(memberLocation) << "a field of a @" CPP_VTABLE " class can't have an initializer";
                }

                if (hasModifier(classMember, SyntaxKind::AccessorKeyword))
                {
                    error(memberLocation) << "a @" CPP_VTABLE " class can't have accessor fields";
                }

                continue;
            }

            if (classMember == SyntaxKind::Constructor)
            {
                if (!implementation)
                {
                    error(memberLocation) << "a @" CPP_VTABLE " class can't have a constructor: its objects are made by C++, "
                                          << "unless its methods have bodies";
                }

                continue;
            }

            // accessors are generated methods: their slots wouldn't follow the declaration order
            error(memberLocation) << "a @" CPP_VTABLE " class can only have methods and fields";
        }

        return failed ? mlir::failure() : mlir::success();
    }

    // @slot(N) of a class member: N; -1 without @slot, -2 when its argument isn't a slot number
    int MLIRGenImpl::getCppSlot(ClassElement classMember, const GenContext &genContext)
    {
        auto slot = -1;
        iterateDecorators(classMember, genContext, [&](StringRef name, SmallVector<StringRef> args) {
            if (name != CPP_SLOT)
            {
                return;
            }

            unsigned value;
            if (args.size() != 1 || args.front().getAsInteger(10, value) || value > (unsigned)std::numeric_limits<int>::max())
            {
                slot = -2;
                return;
            }

            slot = static_cast<int>(value);
        });

        return slot;
    }

    // Whether a base class of a @cpp class has a virtual method named methodName: a method of the
    // class with that name overrides it, as in C++, without @slot of its own
    bool MLIRGenImpl::isCppBaseSlot(ClassInfo::TypePtr classInfo, StringRef methodName)
    {
        for (auto &base : classInfo->baseClasses)
        {
            if (llvm::any_of(base->methods, [&](auto &method) { return method.name == methodName && method.isVirtual && !method.isStatic; })
                || isCppBaseSlot(base, methodName))
            {
                return true;
            }
        }

        return false;
    }

    std::string MLIRGenImpl::getCppMethodName(ClassElement classMember, const GenContext &genContext)
    {
        std::string methodName;
        std::string propertyName;
        if (mlir::failed(getMethodNameOrPropertyName(false, classMember.as<SignatureDeclarationBase>(), methodName, propertyName, genContext)))
        {
            return std::string();
        }

        return methodName;
    }

    // A method of a @cpp class is a C++ virtual method (a vtable slot) with @slot(N), when abstract,
    // or when it overrides a base class's slot by name; otherwise it isn't virtual
    bool MLIRGenImpl::isCppVirtualMethod(ClassInfo::TypePtr classInfo, ClassElement classMember, const GenContext &genContext)
    {
        if (classMember != SyntaxKind::MethodDeclaration || hasModifier(classMember, SyntaxKind::StaticKeyword))
        {
            return false;
        }

        return getCppSlot(classMember, genContext) != -1 || hasModifier(classMember, SyntaxKind::AbstractKeyword)
            || isCppBaseSlot(classInfo, getCppMethodName(classMember, genContext));
    }

    // A @cpp class implements C++ methods when one of its virtual methods has a body; a body of a
    // method that isn't virtual is TypeScript's, and leaves the class a binding to C++ objects
    bool MLIRGenImpl::isCppVTableImplementation(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr classInfo,
                                                const GenContext &genContext)
    {
        return llvm::any_of(classDeclarationAST->members, [&](ClassElement member) {
            return member == SyntaxKind::MethodDeclaration && member.as<MethodDeclaration>()->body
                && isCppVirtualMethod(classInfo, member, genContext);
        });
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassCheckIfDeclaration(mlir::Location location,
                                                       ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        if (declarationMode)
        {
            newClassPtr->isDeclaration = true;
            return mlir::success();
        }

        if (classDeclarationAST != SyntaxKind::ClassExpression)
        {
            return mlir::success();
        }

        for (auto &classMember : classDeclarationAST->members)
        {
            // TODO:
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                // property declaration
                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (propertyDeclaration->initializer)
                {
                    // no definition
                    return mlir::success();
                }
            }

            if (classMember == SyntaxKind::MethodDeclaration || classMember == SyntaxKind::Constructor ||
                classMember == SyntaxKind::GetAccessor || classMember == SyntaxKind::SetAccessor)
            {
                auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
                if (funcLikeDeclaration->body)
                {
                    // no definition
                    return mlir::success();
                }
            }
        }

        newClassPtr->isDeclaration = true;

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeSetFields(ClassInfo::TypePtr newClassPtr,
                                                  SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        if (newClassPtr)
        {
            mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType()).setFields(fieldInfos);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStorageType(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // add base classes
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClause(classDeclarationAST, newClassPtr, heritageClause, fieldInfos,
                                                        genContext)))
            {
                return mlir::failure();
            }
        }

        // with the base classes known: a method with a body implements a C++ method when it
        // overrides a base's slot, which the guess made with the class (any method with a body)
        // couldn't tell from a method of TypeScript's own
        if (newClassPtr->isCppVTable && newClassPtr->isCppVTableImplementation
            && !isCppVTableImplementation(classDeclarationAST, newClassPtr, genContext))
        {
            newClassPtr->isCppVTableImplementation = false;
            newClassPtr->isDeclaration = true;
        }

#if ENABLE_RTTI
        if (newClassPtr->isCppVTable || newClassPtr->isValue)
        {
            // no RTTI: a C++ object can't answer `instanceof`, nor can a value, which has no vtable
        }
        else if (newClassPtr->isDynamicImport)
        {
            mlirGenCustomRTTIDynamicImport(location, classDeclarationAST, newClassPtr, genContext);
        }
        else if (!newClassPtr->isStatic)
        {
            newClassPtr->hasVirtualTable = true;
            mlirGenCustomRTTI(location, classDeclarationAST, newClassPtr, genContext);
        }
#endif

        if (!newClassPtr->isStatic && (!newClassPtr->isCppVTable || newClassPtr->isCppVTableImplementation))
        {
            mlirGenClassSizeStaticField(location, classDeclarationAST, newClassPtr, genContext);
        }

        // non-static first
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, false, genContext)))
            {
                return mlir::failure();
            }
        }

        mlirGenClassGetterOnlyForBase(newClassPtr, classDeclarationAST);

        if (newClassPtr->getHasVirtualTableVariable())
        {
            auto fieldId = MLIRHelper::TupleFieldName(VTABLE_NAME, builder.getContext());
            if (fieldInfos.size() == 0 || fieldInfos.front().id != fieldId)
            {
                fieldInfos.insert(fieldInfos.begin(), {fieldId, getOpaqueType(), false, mlir_ts::AccessLevel::Public});
            }
        }

        mlirGenClassTypeSetFields(newClassPtr, fieldInfos);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFields(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // dummy class, not used, needed to sync code
        // TODO: refactor it
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // static second
        // TODO: if I use static method in static field initialization, test if I need process static fields after
        // static methods
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, true, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMembers(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                            ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // clear all flags
        // extra fields - first, we need .instanceOf first for typr Any

        // dummy class, not used, needed to sync code
        // TODO: refactor it
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // process indexes first
        for (auto &classMember : classDeclarationAST->members)
        {
            if (classMember == SyntaxKind::IndexSignature)
            {
                if (mlir::failed(mlirGenClassIndexMember(newClassPtr, classMember, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            LLVM_DEBUG(llvm::dbgs() << "\n****** \tclass members: " << newClassPtr->fullName << " not resolved: " << notResolved;);

            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            auto orderWeight = 0;
            for (auto &classMember : newClassPtr->extraMembers)
            {
                orderWeight++;
                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    notResolved++;
                }
            }

            for (auto &classMember : classDeclarationAST->members)
            {
                orderWeight++;

                // DEBUG ON
                LLVM_DEBUG(ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);\
                    auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();\
                    getMethodNameOrPropertyName(\
                        newClassPtr->isStatic,\
                        funcLikeDeclaration,\
                        classMethodMemberInfo.methodName,\
                        classMethodMemberInfo.propertyName,\
                        genContext);\
                llvm::dbgs() << "\n****** \tprocessing: " << newClassPtr->fullName << "." << classMethodMemberInfo.methodName;);

                // static fields
                if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, true, genContext)))
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n\tNOT RESOLVED FIELD.");
                    notResolved++;
                }

                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    LLVM_DEBUG(ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);\
                        auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();\
                        getMethodNameOrPropertyName(\
                            newClassPtr->isStatic,\
                            funcLikeDeclaration,\
                            classMethodMemberInfo.methodName,\
                            classMethodMemberInfo.propertyName,\
                            genContext);\
                        llvm::dbgs() << "\n\tNOT RESOLVED MEMBER: " << classMethodMemberInfo.methodName;);
                    notResolved++;
                }

                if (mlir::failed(mlirGenClassStaticBlockMember(classDeclarationAST, newClassPtr, classMember, genContext)))
                {
                    return mlir::failure();
                }
            }

            for (auto &classMember : newClassPtr->extraMembersPost)
            {
                orderWeight++;

                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    notResolved++;
                }
            }            

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depend on other class declarations
                // theModule.emitError("can't resolve dependencies in class: ") << newClassPtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        clearMembersProcessStates(classDeclarationAST, newClassPtr);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassHeritageClause(ClassLikeDeclaration classDeclarationAST,
                                                   ClassInfo::TypePtr newClassPtr, HeritageClause heritageClause,
                                                   SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        if (heritageClause->token == SyntaxKind::ExtendsKeyword)
        {
            auto &baseClassInfos = newClassPtr->baseClasses;

            for (auto &extendingType : heritageClause->types)
            {
                auto result = mlirGen(extendingType, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto baseType = V(result);
                auto success = false;
                mlir::TypeSwitch<mlir::Type>(baseType.getType())
                    .template Case<mlir_ts::ClassType>([&](auto baseClassType) {
                        auto baseName = baseClassType.getName().getValue();
                        auto fieldId = MLIRHelper::TupleFieldName(baseName, builder.getContext());
                        fieldInfos.push_back({fieldId, baseClassType.getStorageType(), false, mlir_ts::AccessLevel::Public});

                        auto classInfo = getClassInfoByFullName(baseName);
                        // a value has no base layout, and nothing extends one: no vtable to override
                        // through
                        if (newClassPtr->isValue)
                        {
                            emitError(loc(extendingType)) << "@" VALUE_DECORATOR " class '" << newClassPtr->name
                                                          << "' cannot extend '" << (classInfo ? classInfo->name : baseName) << "'";
                            return;
                        }

                        if (classInfo && classInfo->isValue)
                        {
                            emitError(loc(extendingType)) << "class '" << newClassPtr->name << "' cannot extend '"
                                                          << classInfo->name << "': it is @" VALUE_DECORATOR;
                            return;
                        }

                        // calls on a @final class are direct (ClassMethodAccess): an override
                        // would never run
                        if (classInfo && classInfo->isFinal)
                        {
                            emitError(loc(extendingType)) << "class '" << newClassPtr->name << "' cannot extend '"
                                                          << classInfo->name << "': it is @" FINAL_DECORATOR;
                            return;
                        }

                        if (std::find(baseClassInfos.begin(), baseClassInfos.end(), classInfo) == baseClassInfos.end())
                        {
                            baseClassInfos.push_back(classInfo);
                        }

                        success = true;
                    })
                    .Default([&](auto type) {
                        // e.g. `class Sub extends SomeInterface {}` - real TypeScript rejects
                        // extending an interface (only a class/constructor value is allowed
                        // there) but this compiler doesn't enforce that rule at the heritage-
                        // clause level, so it reaches here instead of being caught earlier.
                        // Confirmed via repro. The original code had no failure-tracking at all
                        // here (unconditionally `return mlir::success()` below regardless of
                        // which case fired) - added `success` so this diagnostic isn't silently
                        // swallowed the way it was on first attempt (emitError alone doesn't
                        // fail this function; verified the fix actually surfaces as a compile
                        // error, not just a silently-succeeding no-op).
                        emitError(loc(extendingType)) << "class cannot extend type: " << to_print(type);
                    });

                if (!success)
                {
                    return mlir::failure();
                }
            }
            return mlir::success();
        }

        if (heritageClause->token == SyntaxKind::ImplementsKeyword)
        {
            // a value has no vtable to see it through an interface by
            if (newClassPtr->isValue)
            {
                emitError(loc(heritageClause)) << "@" VALUE_DECORATOR " class '" << newClassPtr->name
                                               << "' cannot implement an interface";
                return mlir::failure();
            }

            newClassPtr->hasVirtualTable = true;

            auto &interfaceInfos = newClassPtr->implements;

            for (auto &implementingType : heritageClause->types)
            {
                auto result = mlirGen(implementingType, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto ifaceType = V(result);
                mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                    .template Case<mlir_ts::InterfaceType>([&](mlir_ts::InterfaceType interfaceType) {

                        auto ifaceName = interfaceType.getName().getValue();
                        auto found = llvm::find_if(interfaceInfos, [&](ImplementInfo &ifaceInfo) {
                            return ifaceInfo.interface->fullName == ifaceName;
                        });

                        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                        assert(interfaceInfo);
                        if (found != interfaceInfos.end()) {
                            found->interface = interfaceInfo;
                        } else {
                            interfaceInfos.push_back({interfaceInfo, -1, false});
                        }
                    })
                    .Default([&](auto type) {
                        // e.g. `class X implements SomeTypeAlias` where the alias resolves to an
                        // object-shape TupleType (`type Shape = {area(): number}`) rather than a
                        // declared InterfaceType - real, valid, and common TypeScript (implementing
                        // a type-alias-described shape), confirmed via repro. This compiler's
                        // `implements` handling only accepts a genuine InterfaceType - properly
                        // supporting a TupleType target would need synthesizing an anonymous
                        // interface from it, a separate, scoped feature. Fail cleanly for now.
                        emitError(loc(implementingType)) << "class cannot implement type: " << to_print(type);
                    });
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDataFieldAccessor(mlir::Location location, ClassInfo::TypePtr newClassPtr, 
            PropertyDeclaration propertyDeclaration, MemberName name, mlir::Type typeIfNotProvided, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        NodeArray<ModifierLike> modifiers;
        for (auto modifier : propertyDeclaration->modifiers)
        {
            if (modifier == SyntaxKind::AccessorKeyword)
            {
                continue;
            }

            modifiers.push_back(modifier);
        }

        // add accessor methods
        if ((propertyDeclaration->internalFlags & InternalFlags::GenerationProcessed) != InternalFlags::GenerationProcessed)
        {            
            // set as generated
            propertyDeclaration->internalFlags |= InternalFlags::GenerationProcessed;

            {
                NodeArray<Statement> statements;

                auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

                auto propAccess = nf.createPropertyAccessExpression(thisToken, name);

                auto returnStat = nf.createReturnStatement(propAccess);
                statements.push_back(returnStat);

                auto body = nf.createBlock(statements, /*multiLine*/ false);

                auto getMethod = nf.createGetAccessorDeclaration(modifiers, propertyDeclaration->name, {}, undefined, body);

                newClassPtr->extraMembersPost->push_back(getMethod);
            }

            {
                NodeArray<Statement> statements;

                auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

                auto propAccess = nf.createPropertyAccessExpression(thisToken, name);

                auto setValue =
                    nf.createExpressionStatement(
                        nf.createBinaryExpression(propAccess, nf.createToken(SyntaxKind::EqualsToken), nf.createIdentifier(S("value"))));
                statements.push_back(setValue);

                auto body = nf.createBlock(statements, /*multiLine*/ false);

                auto type = propertyDeclaration->type;
                if (!type && typeIfNotProvided)
                {
                    std::string fieldTypeAlias;
                    fieldTypeAlias += ".";
                    fieldTypeAlias += newClassPtr->fullName.str();
                    fieldTypeAlias += ".";
                    fieldTypeAlias += MLIRHelper::getName(name);
                    type = nf.createTypeReferenceNode(nf.createIdentifier(stows(fieldTypeAlias)), undefined);    

                    getTypeAliasMap().insert({fieldTypeAlias, { typeIfNotProvided, undefined }});
                }

                if (!type)
                {
                    emitError(location) << "type for field accessor '" << MLIRHelper::getName(propertyDeclaration->name) << "' must be provided";
                    return mlir::failure();
                }

                auto setMethod = nf.createSetAccessorDeclaration(
                    modifiers, 
                    propertyDeclaration->name, 
                    { nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("value")), undefined, type) }, 
                    body);

                newClassPtr->extraMembersPost->push_back(setMethod);
            }
        }        

        return mlir::success();
    }

    // Accessors made for a field's name (mlirGenClassAbstractFieldAccessors, mlirGenClassFieldAccessorsForBase):
    // abstract ones with no body, or ones reading and writing the class's field of that name - a member
    // access finds a field before an accessor, so `this.prop` in their bodies is the field. A class is
    // processed more than once; they are added once.
    void MLIRGenImpl::mlirGenClassSynthAccessors(ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration,
                                                 mlir::Type type, bool getter, bool setter, bool isAbstract)
    {
        auto name = propertyDeclaration->name;
        auto nameStr = MLIRHelper::getName(name);
        if (llvm::any_of(newClassPtr->extraMembersPost, [&](ClassElement member) {
                return (member == SyntaxKind::GetAccessor || member == SyntaxKind::SetAccessor)
                    && MLIRHelper::getName(member.as<AccessorDeclaration>()->name) == nameStr;
            }))
        {
            return;
        }

        NodeFactory nf(NodeFactoryFlags::None);

        auto modifiers = [&]() {
            NodeArray<ModifierLike> modifiers;
            for (auto modifier : propertyDeclaration->modifiers)
            {
                if (modifier == SyntaxKind::PublicKeyword || modifier == SyntaxKind::ProtectedKeyword
                    || modifier == SyntaxKind::PrivateKeyword)
                {
                    modifiers.push_back(modifier);
                }
            }

            if (isAbstract)
            {
                modifiers.push_back(nf.createToken(SyntaxKind::AbstractKeyword));
            }

            return modifiers;
        };

        // the type is known as an mlir type only: named by an alias, as mlirGenClassDataFieldAccessor does
        auto typeAlias = StringRef("." + newClassPtr->fullName.str() + "." + nameStr + ".accessor").copy(stringAllocator);
        getTypeAliasMap()[typeAlias] = { type, undefined };
        auto typeNode = [&]() { return nf.createTypeReferenceNode(nf.createIdentifier(stows(typeAlias.str())), undefined); };

        auto thisField = [&]() { return nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), name); };

        if (getter)
        {
            ts::Block body;
            if (!isAbstract)
            {
                NodeArray<Statement> statements;
                statements.push_back(nf.createReturnStatement(thisField()));
                body = nf.createBlock(statements, /*multiLine*/ false);
            }

            newClassPtr->extraMembersPost.push_back(nf.createGetAccessorDeclaration(modifiers(), name, {}, typeNode(), body));
        }

        if (setter)
        {
            ts::Block body;
            if (!isAbstract)
            {
                NodeArray<Statement> statements;
                statements.push_back(nf.createExpressionStatement(nf.createBinaryExpression(
                    thisField(), nf.createToken(SyntaxKind::EqualsToken), nf.createIdentifier(S("value")))));
                body = nf.createBlock(statements, /*multiLine*/ false);
            }

            newClassPtr->extraMembersPost.push_back(nf.createSetAccessorDeclaration(
                modifiers(), name,
                {nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("value")), undefined, typeNode())},
                body));
        }
    }

    // `abstract prop: T` has no storage: it is `abstract get prop(): T` and `abstract set prop(value: T)` (no
    // setter when readonly), virtual as accessors are, so a read or write through the base reaches whatever a
    // subclass implements it with - accessors of its own, or a field (mlirGenClassFieldAccessorsForBase).
    mlir::LogicalResult MLIRGenImpl::mlirGenClassAbstractFieldAccessors(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                                        PropertyDeclaration propertyDeclaration,
                                                                        const GenContext &genContext)
    {
        auto [type, init] = mlirGenClassDataFieldType(location, propertyDeclaration, genContext);
        if (mth.isNoneType(type))
        {
            if (genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            emitError(location) << "type for abstract field '" << MLIRHelper::getName(propertyDeclaration->name)
                                << "' must be provided";
            return mlir::failure();
        }

        mlirGenClassSynthAccessors(newClassPtr, propertyDeclaration, type, true,
                                   !hasModifier(propertyDeclaration, SyntaxKind::ReadonlyKeyword), true);
        return mlir::success();
    }

    // the numeric result of the nearest base class's method (or accessor) of the name, which a method of the same
    // name without a return type overrides: `f() { return 3; }` infers s32, but its slot is called as returning
    // the base's `number`
    mlir::Type MLIRGenImpl::overriddenNumericReturnType(ClassInfo::TypePtr classInfo, StringRef methodName)
    {
        for (auto &base : classInfo->baseClasses)
        {
            auto index = base->getMethodIndex(methodName);
            if (index >= 0)
            {
                auto funcType = base->methods[index].funcType;
                return funcType && funcType.getNumResults() > 0 && mth.isNumericType(funcType.getResult(0))
                    ? funcType.getResult(0)
                    : mlir::Type();
            }

            if (auto found = overriddenNumericReturnType(base, methodName))
            {
                return found;
            }
        }

        return mlir::Type();
    }

    // the nearest base class's accessor of the name: one implemented on the way is not abstract
    static AccessorInfo *nearestBaseAccessor(ClassInfo::TypePtr classInfo, StringRef name)
    {
        for (auto &base : classInfo->baseClasses)
        {
            auto index = base->getAccessorIndex(name);
            if (index >= 0)
            {
                return &base->accessors[index];
            }

            if (auto *found = nearestBaseAccessor(base, name))
            {
                return found;
            }
        }

        return nullptr;
    }

    // A field implementing an abstract accessor of a base class (`abstract prop: T` there, or
    // `abstract get prop(): T`): the class gets accessors over the field, which fill the base's slots, so a
    // read through the base reads the field. Through the class itself the field is read, found first.
    void MLIRGenImpl::mlirGenClassFieldAccessorsForBase(ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration)
    {
        auto nameStr = MLIRHelper::getName(propertyDeclaration->name);
        auto *accessor = nearestBaseAccessor(newClassPtr, nameStr);
        if (!accessor || accessor->isStatic || !accessor->isAbstract)
        {
            return;
        }

        auto type = accessor->get ? accessor->get.funcType.getResult(0) : accessor->set.funcType.getInput(1);
        mlirGenClassSynthAccessors(newClassPtr, propertyDeclaration, type, !!accessor->get, !!accessor->set, false);
    }

    // A getter alone implementing an abstract accessor pair of a base class (`abstract prop: T` there): a write
    // through the base fills the setter's slot, so the class gets a setter that throws, as JavaScript's write to a
    // property with only a getter does - a TypeError, or a failing assert without the default library.
    void MLIRGenImpl::mlirGenClassGetterOnlyForBase(ClassInfo::TypePtr newClassPtr, ClassLikeDeclaration classDeclarationAST)
    {
        for (auto &classMember : classDeclarationAST->members)
        {
            if (classMember != SyntaxKind::GetAccessor || hasModifier(classMember, SyntaxKind::StaticKeyword))
            {
                continue;
            }

            auto name = classMember.as<AccessorDeclaration>()->name;
            auto nameStr = MLIRHelper::getName(name);
            auto hasMember = [&](NodeArray<ClassElement> members, SyntaxKind kind) {
                return llvm::any_of(members, [&](ClassElement member) {
                    return member == kind && MLIRHelper::getName(member.as<AccessorDeclaration>()->name) == nameStr;
                });
            };

            if (hasMember(classDeclarationAST->members, SyntaxKind::SetAccessor)
                || hasMember(newClassPtr->extraMembersPost, SyntaxKind::SetAccessor))
            {
                continue;
            }

            auto *accessor = nearestBaseAccessor(newClassPtr, nameStr);
            if (!accessor || accessor->isStatic || !accessor->isAbstract || !accessor->set)
            {
                continue;
            }

            NodeFactory nf(NodeFactoryFlags::None);

            auto typeAlias = StringRef("." + newClassPtr->fullName.str() + "." + nameStr + ".accessor").copy(stringAllocator);
            getTypeAliasMap()[typeAlias] = { accessor->set.funcType.getInput(1), undefined };

            auto message = nf.createStringLiteral(
                stows("Cannot set property " + nameStr + " of " + newClassPtr->name.str() + " which has only a getter"));

            NodeArray<Statement> statements;
            if (getClassInfoByFullName("TypeError"))
            {
                NodeArray<Expression> arguments;
                arguments.push_back(message);
                statements.push_back(nf.createThrowStatement(
                    nf.createNewExpression(nf.createIdentifier(S("TypeError")), undefined, arguments)));
            }
            else
            {
                NodeArray<Expression> arguments;
                arguments.push_back(nf.createToken(SyntaxKind::FalseKeyword));
                arguments.push_back(message);
                statements.push_back(nf.createExpressionStatement(
                    nf.createCallExpression(nf.createIdentifier(S("assert")), undefined, arguments)));
            }

            NodeArray<ModifierLike> modifiers;
            for (auto modifier : classMember.as<AccessorDeclaration>()->modifiers)
            {
                if (modifier == SyntaxKind::PublicKeyword || modifier == SyntaxKind::ProtectedKeyword
                    || modifier == SyntaxKind::PrivateKeyword)
                {
                    modifiers.push_back(modifier);
                }
            }

            newClassPtr->extraMembersPost.push_back(nf.createSetAccessorDeclaration(
                modifiers, name,
                {nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("value")), undefined,
                                               nf.createTypeReferenceNode(nf.createIdentifier(stows(typeAlias.str())), undefined))},
                nf.createBlock(statements, /*multiLine*/ false)));
        }
    }

    std::pair<mlir::Type, bool> MLIRGenImpl::mlirGenClassDataFieldType(mlir::Location location, PropertyDeclaration propertyDeclaration,
                                                                     const GenContext &genContext)
    {
        auto [type, init, typeProvided] = evaluateTypeAndInit(propertyDeclaration, genContext);
        if (init)
        {
            type = mth.wideStorageType(type);
        }

        // a field's type inferred from `fn = () => 0` is `() => number`, which a capturing closure
        // can be assigned to later: store it as a hybrid function, as an annotated field and a `let`
        // (adjustLocalVariableType) are, not as a plain one with no room for the closure's context
        if (!typeProvided)
        {
            if (auto funcType = dyn_cast_or_null<mlir_ts::FunctionType>(type); funcType && !funcType.isCVarArg())
            {
                type = mlir_ts::HybridFunctionType::get(builder.getContext(), funcType);
            }
        }

        // `v?: T` is `v: T | undefined`, as in a type literal: unassigned it reads as undefined, not as T's zero.
        // An interface's `v?: T` member reads it by its flag (see mlirGenClassVirtualTableDefinitionForInterface).
        if (propertyDeclaration->questionToken && type && !mth.isNoneType(type))
        {
            type = getUnionType(location, type, getUndefinedType());
        }

        return {type, init};
    }

    // the fields mlirGenClassDataFieldMember left untyped in a discovery pass get their types now that
    // the members are declared; code made before against the placeholder is thrown away with the pass
    mlir::LogicalResult MLIRGenImpl::mlirGenClassRetypeFields(ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        if (newClassPtr->fieldsTypedAfterMembers.empty())
        {
            return mlir::success();
        }

        auto storageType = mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType());
        SmallVector<mlir_ts::FieldInfo> fieldInfos(storageType.getFields().begin(), storageType.getFields().end());
        for (auto propertyDeclaration : newClassPtr->fieldsTypedAfterMembers)
        {
            auto [type, init] = mlirGenClassDataFieldType(loc(propertyDeclaration), propertyDeclaration, genContext);
            if (mth.isNoneType(type))
            {
                continue;
            }

            auto fieldId = TupleFieldName(propertyDeclaration->name, genContext);
            for (auto &fieldInfo : fieldInfos)
            {
                if (fieldInfo.id == fieldId)
                {
                    fieldInfo.type = type;
                }
            }
        }

        newClassPtr->fieldsTypedAfterMembers.clear();
        return storageType.setFields(fieldInfos);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDataFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                    PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto accessLevel = getAccessLevel(propertyDeclaration);

        if (hasModifier(propertyDeclaration, SyntaxKind::AbstractKeyword))
        {
            return mlirGenClassAbstractFieldAccessors(location, newClassPtr, propertyDeclaration, genContext);
        }

        auto name = propertyDeclaration->name;
        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            name = getFieldNameForAccessor(name);
        }

        auto fieldId = TupleFieldName(name, genContext);
        if (!fieldId)
        {
            // e.g. a computed field name (`[key] = 42;`) that doesn't resolve to a compile-time
            // constant (getNameFromComputedPropertyName already emitted its own diagnostic in
            // that case) - fail cleanly here instead of crashing on the null Attribute below
            // (dyn_cast/cast require their input to be non-null; that's what dyn_cast_or_null is
            // for, and this code wasn't using it).
            return mlir::failure();
        }

        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            if (strAttr.getValue().starts_with("#"))
            {
                accessLevel = mlir_ts::AccessLevel::Private;
            }
        }

        // an instance field: a static one is a global, not part of the value
        auto layoutField = valueLayoutField(newClassPtr->isValue, newClassPtr->fullName, newClassPtr->name, location,
                                            MLIRHelper::getName(name));
        auto [type, init] = mlirGenClassDataFieldType(location, propertyDeclaration, genContext);
        if (init)
        {
            newClassPtr->hasInitializers = true;
        }

        auto hasType = !!propertyDeclaration->type;

        LLVM_DEBUG(dbgs() << "\n!! class field: " << fieldId << " type: " << type << " access level: " << accessLevel);
        if (mth.isNoneType(type))
        {
            if (hasType)
            {
                return mlir::failure();
            }

            // `fn = this.m`: a discovery pass lays out the fields before it declares the methods, so
            // the initializer has no type yet. Hold the place and type the field once they are.
            if (init && !isAccessor && genContext.allowPartialResolve)
            {
                newClassPtr->fieldsTypedAfterMembers.push_back(propertyDeclaration);
                fieldInfos.push_back({fieldId, getAnyType(), false, accessLevel});
                return mlir::success();
            }

#ifndef ANY_AS_DEFAULT
            emitError(location)
                << "type for field '" << fieldId << "' is not provided, field must have type or initializer";
            return mlir::failure();
#else
            emitWarning(location) << "type for field '" << fieldId << "' is any";
            type = getAnyType();
#endif
        }

        fieldInfos.push_back({fieldId, type, false, accessLevel});

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, type, genContext);
            EXIT_IF_FAILED(res)
        }
        else
        {
            mlirGenClassFieldAccessorsForBase(newClassPtr, propertyDeclaration);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto accessLevel = getAccessLevel(propertyDeclaration);
        auto isPublic = accessLevel == mlir_ts::AccessLevel::Public;
        auto name = propertyDeclaration->name;

        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            isPublic = false;
            name = getFieldNameForAccessor(name);
        }

        auto fieldId = TupleFieldName(name, genContext);
        if (!fieldId)
        {
            // same shape as mlirGenClassDataFieldMember's identical guard above - a computed
            // field name that failed to resolve to a constant already got its own diagnostic
            // from getNameFromComputedPropertyName; fail cleanly instead of crashing on the
            // unconditional mlir::cast<StringAttr> a few lines below.
            return mlir::failure();
        }

        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            if (strAttr.getValue().starts_with("#"))
            {
                isPublic = false;
                accessLevel = mlir_ts::AccessLevel::Private;
            }
        }

        // process static field - register global
        auto fullClassStaticFieldName =
            concat(newClassPtr->fullName, mlir::cast<mlir::StringAttr>(fieldId).getValue());
        VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
        varClass.isExport = newClassPtr->isExport && isPublic;
        varClass.isImport = newClassPtr->isImport && isPublic;
        varClass.isPublic = isPublic;

        auto staticFieldType = registerVariable(
            location, fullClassStaticFieldName, true, varClass,
            [&](mlir::Location location, const GenContext &genContext) {
                auto isConst = false;
                mlir::Type typeInit;
                evaluate(
                    propertyDeclaration->initializer,
                    [&](mlir::Value val) {
                        typeInit = val.getType();
                        typeInit = mth.wideStorageType(typeInit);
                        isConst = isConstValue(val);
                    },
                    genContext);

                if (!newClassPtr->isDeclaration)
                {
                    if (isConst)
                    {
                        return getTypeAndInit(propertyDeclaration, genContext);
                    }

                    newClassPtr->hasStaticInitializers = true;
                }

                return getTypeOnly(propertyDeclaration, typeInit, genContext);
            },
            genContext);

        // a failed initializer (e.g. `null` for a `Ref<string>`) leaves no type; going on with
        // none dropped its diagnostic, since those are only shown when the module fails
        if (!staticFieldType)
        {
            return mlir::failure();
        }

        auto &staticFieldInfos = newClassPtr->staticFields;
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, accessLevel);

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, staticFieldType, genContext);
            EXIT_IF_FAILED(res)
        }  

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFieldMemberDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto fieldId = TupleFieldName(propertyDeclaration->name, genContext);
        if (!fieldId)
        {
            // same shape as mlirGenClassDataFieldMember's/mlirGenClassStaticFieldMember's
            // identical guard - fail cleanly instead of crashing on the unconditional
            // mlir::cast<StringAttr> below.
            return mlir::failure();
        }

        auto accessLevel = getAccessLevel(propertyDeclaration);

        // process static field - register global
        auto fullClassStaticFieldName =
            concat(newClassPtr->fullName, mlir::cast<mlir::StringAttr>(fieldId).getValue());

        auto staticFieldType = registerVariable(
            location, fullClassStaticFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) -> TypeValueInitType {
                // detect field Type
                auto isConst = false;
                mlir::Type typeInit;
                if (propertyDeclaration->type)
                {
                    typeInit = getType(propertyDeclaration->type, genContext);
                }
                else if (propertyDeclaration->initializer)
                {
                    evaluate(
                        propertyDeclaration->initializer,
                        [&](mlir::Value val) {
                            typeInit = val.getType();
                            typeInit = mth.wideStorageType(typeInit);
                            isConst = isConstValue(val);
                        },
                        genContext);
                }
                else
                {
                    return {mlir::Type(), mlir::Value(), TypeProvided::No};
                }

                // add command to load reference from DLL
                auto fullName = V(mlirGenStringValue(location, fullClassStaticFieldName.str(), true));
                auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, getOpaqueType(), fullName);
                auto result = cast(location, mlir_ts::RefType::get(typeInit), referenceToStaticFieldOpaque, genContext);
                auto referenceToStaticField = V(result);
                return {referenceToStaticField.getType(), referenceToStaticField, TypeProvided::No};
            },
            genContext);

        if (!staticFieldType)
        {
            return mlir::failure();
        }

        auto &staticFieldInfos = newClassPtr->staticFields;
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, accessLevel);

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenClassConstructorPublicDataFieldMembers(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                                      SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                                      ConstructorDeclaration constructorDeclaration, const GenContext &genContext)
    {
        for (auto &parameter : constructorDeclaration->parameters)
        {
            auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
            auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
            auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

            if (!(isPublic || isProtected || isPrivate))
            {
                continue;
            }

            auto fieldId = TupleFieldName(parameter->name, genContext);
            if (!fieldId)
            {
                // constructor parameter names can't actually be computed property names (only a
                // plain identifier or binding pattern), so this is believed unreachable in
                // practice - guarded anyway for consistency with this function's siblings above,
                // since dyn_cast (unlike dyn_cast_or_null) still asserts on a null Attribute.
                return mlir::failure();
            }

            if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId)) {
                isPrivate |= strAttr.getValue().starts_with("#");
            }

            // a field of another constructor overload's parameter property already
            if (llvm::any_of(fieldInfos, [&](mlir_ts::FieldInfo &fieldInfo) { return fieldInfo.id == fieldId; }))
            {
                continue;
            }

            auto layoutField = valueLayoutField(newClassPtr->isValue, newClassPtr->fullName, newClassPtr->name, loc(parameter),
                                                MLIRHelper::getName(parameter->name));
            auto [type, init, typeProvided] = evaluateTypeAndInit(parameter, genContext);

            if (mth.isNoneType(type))
            {
                return mlir::failure();
            }

            // `constructor(public m?: T)` is a field `m?: T`: T | undefined, as a declared one
            // (mlirGenClassDataFieldMember) - `new Error().message` is undefined
            if (parameter->questionToken)
            {
                type = getUnionType(location, type, getUndefinedType());
            }

            LLVM_DEBUG(dbgs() << "\n+++ class auto-gen field: " << fieldId << " type: " << type << "");

            fieldInfos.push_back(
            {
                fieldId, 
                type, 
                false, 
                isPrivate 
                    ? mlir_ts::AccessLevel::Private 
                    : isProtected 
                        ? mlir_ts::AccessLevel::Protected 
                        : mlir_ts::AccessLevel::Public 
            });
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassProcessClassPropertyByFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember)
    {
        auto isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
        auto isConstructor = classMember == SyntaxKind::Constructor;
        if (isConstructor)
        {
            if (isStatic)
            {
                newClassPtr->hasStaticConstructor = true;
            }
            else
            {
                newClassPtr->hasConstructor = true;
                newClassPtr->constructorAccessLevel = getAccessLevel(classMember);
            }
        }

        // a @value class has no vtable: its methods are called directly
        if (newClassPtr->isStatic || newClassPtr->isValue)
        {
            return mlir::success();
        }

        auto isMemberAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        if (isMemberAbstract)
        {
            newClassPtr->hasVirtualTable = true;
        }

        auto isVirtual = (classMember->internalFlags & InternalFlags::ForceVirtual) == InternalFlags::ForceVirtual;
#ifdef ALL_METHODS_VIRTUAL
        isVirtual = !isConstructor;
#endif
        if (isVirtual)
        {
            newClassPtr->hasVirtualTable = true;
        }

        return mlir::success();
    }

    // An instance member of a @value class: every field `readonly` (a parameter property too), and no
    // setter. Statics are as any class's.
    mlir::LogicalResult MLIRGenImpl::mlirGenValueClassMemberRules(ClassInfo::TypePtr newClassPtr, ClassElement classMember)
    {
        auto fieldMustBeReadonly = [&](Node fieldNode, Node nameNode) {
            emitError(loc(fieldNode)) << "field '" << MLIRHelper::getName(nameNode) << "' of @" VALUE_DECORATOR " class '"
                                      << newClassPtr->name << "' must be readonly";
            return mlir::failure();
        };

        if (classMember == SyntaxKind::PropertyDeclaration)
        {
            auto propertyDeclaration = classMember.as<PropertyDeclaration>();
            if (!hasModifier(propertyDeclaration, SyntaxKind::ReadonlyKeyword))
            {
                return fieldMustBeReadonly(propertyDeclaration, propertyDeclaration->name);
            }
        }

        if (classMember == SyntaxKind::Constructor)
        {
            for (auto &parameter : classMember.as<ConstructorDeclaration>()->parameters)
            {
                auto isProperty = hasModifier(parameter, SyntaxKind::PublicKeyword) || hasModifier(parameter, SyntaxKind::ProtectedKeyword)
                                  || hasModifier(parameter, SyntaxKind::PrivateKeyword) || hasModifier(parameter, SyntaxKind::ReadonlyKeyword);
                if (isProperty && !hasModifier(parameter, SyntaxKind::ReadonlyKeyword))
                {
                    return fieldMustBeReadonly(parameter, parameter->name);
                }
            }
        }

        if (classMember == SyntaxKind::SetAccessor)
        {
            emitError(loc(classMember)) << "@" VALUE_DECORATOR " class '" << newClassPtr->name << "' cannot have a setter";
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                SmallVector<mlir_ts::FieldInfo> &fieldInfos, bool staticOnly,
                                                const GenContext &genContext)
    {
        auto isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
        if (staticOnly != isStatic)
        {
            return mlir::success();
        }

        auto location = loc(classMember);

        mlirGenClassProcessClassPropertyByFieldMember(newClassPtr, classMember);

        // the copies of a @value value must not differ: no field it could change, no setter
        if (newClassPtr->isValue && !isStatic && mlir::failed(mlirGenValueClassMemberRules(newClassPtr, classMember)))
        {
            return mlir::failure();
        }

        if (classMember == SyntaxKind::PropertyDeclaration)
        {
            // property declaration
            auto propertyDeclaration = classMember.as<PropertyDeclaration>();
            if (!isStatic)
            {
                if (mlir::failed(mlirGenClassDataFieldMember(location, newClassPtr, fieldInfos, propertyDeclaration, genContext)))
                {
                    return mlir::failure();
                }
            }
            else
            {
                if (newClassPtr->isDynamicImport)
                {
                    if (mlir::failed(mlirGenClassStaticFieldMemberDynamicImport(location, newClassPtr, propertyDeclaration, genContext)))
                    {
                        return mlir::failure();
                    }
                }
                else if (mlir::failed(mlirGenClassStaticFieldMember(location, newClassPtr, propertyDeclaration, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        if (classMember == SyntaxKind::Constructor && !isStatic)
        {
            auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
            if (mlir::failed(mlirGenClassConstructorPublicDataFieldMembers(location, newClassPtr, fieldInfos, constructorDeclaration, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassNew(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                        const GenContext &genContext)
    {
        if (newClassPtr->isAbstract || newClassPtr->hasNew)
        {
            return mlir::success();
        }

        // create constructor
        newClassPtr->hasNew = true;

        // if we do not have constructor but have initializers we need to create empty dummy constructor
        NodeFactory nf(NodeFactoryFlags::None);

        ts::Block body;
        auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

        if (!newClassPtr->isDeclaration)
        {
            NodeArray<Statement> statements;

            auto newCall = nf.createNewExpression(thisToken, undefined, undefined);
            newCall->internalFlags |= InternalFlags::SuppressConstructorCall;

            auto returnStat = nf.createReturnStatement(newCall);
            statements.push_back(returnStat);

            body = nf.createBlock(statements, /*multiLine*/ false);
        }

        ModifiersArray modifiers;
        modifiers->push_back(nf.createToken(SyntaxKind::StaticKeyword));

        if (newClassPtr->isExport || newClassPtr->isImport)
        {
            modifiers.push_back(nf.createToken(SyntaxKind::PublicKeyword));
        }

        auto generatedNew = nf.createMethodDeclaration(modifiers, undefined, nf.createIdentifier(S(NEW_METHOD_NAME)),
                                                       undefined, undefined, undefined, nf.createThisTypeNode(), body);

        /*
        // advance declaration of "new"
        auto isStatic = false;
#ifdef ALL_METHODS_VIRTUAL
        auto isVirtual = true;
#else
        auto isVirtual = false;
#endif
        SmallVector<mlir::Type> inputs;
        SmallVector<mlir::Type> results{newClassPtr->classType};
        mlirGenForwardDeclaration(NEW_METHOD_NAME, getFunctionType(inputs, results), isStatic, isVirtual, newClassPtr,
genContext);

        newClassPtr->extraMembersPost.push_back(generatedNew);
        */

        LLVM_DEBUG(printDebug(generatedNew););

        newClassPtr->extraMembers.push_back(generatedNew);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasInitializers && !newClassPtr->hasConstructor)
        {
            // create constructor
            newClassPtr->hasConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            // errors in the generated code are reported at the class
            auto atClass = [&](auto node) {
                node->pos = classDeclarationAST->pos;
                node->_end = classDeclarationAST->_end;
                return node;
            };

            NodeArray<ParameterDeclaration> parameters;
            NodeArray<Statement> statements;

            if (!newClassPtr->baseClasses.empty())
            {
                // TypeScript's implicit `constructor(...args) { super(...args); }`: it takes the
                // parameters of the base constructor, typed as its function is (a generic base's
                // already specialized), and passes them on; the base applies its defaults
                mlir_ts::FunctionType baseConstructorType;
                for (auto baseClass = newClassPtr->baseClasses.front(); baseClass;
                     baseClass = baseClass->baseClasses.empty() ? ClassInfo::TypePtr() : baseClass->baseClasses.front())
                {
                    if (baseClass->hasConstructor)
                    {
                        auto index = baseClass->getMethodIndex(CONSTRUCTOR_NAME);
                        if (index >= 0)
                        {
                            baseConstructorType = baseClass->methods[index].funcType;
                        }

                        break;
                    }
                }

                NodeArray<Expression> superArguments;
                if (baseConstructorType)
                {
                    auto inputs = baseConstructorType.getInputs();
                    // the first input is `this`
                    for (auto index = 1u; index < inputs.size(); index++)
                    {
                        auto paramType = inputs[index];
                        auto isRest = baseConstructorType.isVarArg() && index == inputs.size() - 1;
                        auto isOptional = false;
                        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(paramType))
                        {
                            paramType = optionalType.getElementType();
                            isOptional = true;
                        }

                        // not a name the field initializers, run in this constructor, could mean
                        auto paramName = ".arg" + std::to_string(index - 1);

                        std::string paramTypeAlias;
                        paramTypeAlias += ".";
                        paramTypeAlias += newClassPtr->fullName.str();
                        paramTypeAlias += ".";
                        paramTypeAlias += CONSTRUCTOR_NAME;
                        paramTypeAlias += paramName;
                        getTypeAliasMap()[paramTypeAlias] = { paramType, undefined };

                        DotDotDotToken dotDotDotToken;
                        if (isRest)
                        {
                            dotDotDotToken = nf.createToken(SyntaxKind::DotDotDotToken);
                        }

                        QuestionToken questionToken;
                        if (isOptional)
                        {
                            questionToken = nf.createToken(SyntaxKind::QuestionToken);
                        }

                        parameters.push_back(atClass(nf.createParameterDeclaration(
                            undefined, dotDotDotToken, nf.createIdentifier(stows(paramName)), questionToken,
                            nf.createTypeReferenceNode(nf.createIdentifier(stows(paramTypeAlias)), undefined))));

                        Expression argument = atClass(nf.createIdentifier(stows(paramName)));
                        if (isRest)
                        {
                            argument = atClass(nf.createSpreadElement(argument));
                        }

                        superArguments.push_back(argument);
                    }
                }

                auto superExpr = atClass(nf.createToken(SyntaxKind::SuperKeyword));
                auto callSuper = atClass(nf.createCallExpression(superExpr, undefined, superArguments));
                statements.push_back(atClass(nf.createExpressionStatement(callSuper)));
            }

            auto body = atClass(nf.createBlock(statements, /*multiLine*/ false));

            auto generatedConstructor = atClass(nf.createConstructorDeclaration(undefined, parameters, body));
            newClassPtr->extraMembers.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDefaultStaticConstructor(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasStaticInitializers && !newClassPtr->hasStaticConstructor)
        {
            // create constructor
            newClassPtr->hasStaticConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;
            auto body = nf.createBlock(statements, /*multiLine*/ false);
            ModifiersArray modifiers;
            modifiers.push_back(nf.createToken(SyntaxKind::StaticKeyword));
            auto generatedConstructor = nf.createConstructorDeclaration(modifiers, undefined, body);
            newClassPtr->extraMembersPost.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassSizeStaticField(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        auto &staticFieldInfos = newClassPtr->staticFields;

        auto fieldId = MLIRHelper::TupleFieldName(SIZE_NAME, builder.getContext());

        // register global
        auto fullClassStaticFieldName = concat(newClassPtr->fullName, SIZE_NAME);

        auto staticFieldType = getIndexType();

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            // saving state
            auto declarationModeStore = declarationMode;

            // prevent double generating
            //VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
            VariableClass varClass = VariableType::Var;
            varClass.isExport = newClassPtr->isExport;
            varClass.isImport = newClassPtr->isImport;
            varClass.isPublic = true;
            if (!newClassPtr->isImport)
            {                           
                declarationMode = false;
                // decided by the target, not the host: ELF only supports Any (and NoDeduplicate)
                varClass.comdat = compileOptions.isWindows ? Select::ExactMatch : Select::Any;
            }
            else if (newClassPtr->isDeclaration)
            {
                varClass.type = VariableType::External;
            }

            registerVariable(
                location, fullClassStaticFieldName, true, varClass,
                [&](mlir::Location location, const GenContext &genContext) {
                    // if (newClassPtr->isDeclaration)
                    // {
                    //     return std::make_tuple(staticFieldType, mlir::Value(), TypeProvided::Yes);
                    // }

                    // TODO: review usage of SizeOf in code, as size of class pointer is not size of data struct
                    auto sizeOfType =
                        builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), newClassPtr->classType.getStorageType());

                    mlir::Value init = sizeOfType;
                    return std::make_tuple(staticFieldType, init, TypeProvided::Yes);
                },
                genContext);

            // restore state
            declarationMode = declarationModeStore;
        }

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, mlir_ts::AccessLevel::Public);

        return mlir::success();    
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeDescriptorField(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                        const GenContext &genContext)
    {
        // TODO: experiment if we need it at all even external declaration
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        // register global
        auto fullClassStaticFieldName = getTypeDescriptorFieldName(newClassPtr);

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            registerVariable(
                location, fullClassStaticFieldName, true,
                newClassPtr->isDeclaration ? VariableType::External : VariableType::Var,
                [&](mlir::Location location, const GenContext &genContext) {
                    // target-width: the global caches a Boehm GC_descr (= GC_word, the integer the size
                    // of `void *`), so it is pointer-wide rather than a fixed 64 bits.
                    auto typeDescrType = builder.getIntegerType(compileOptions.sizeBits());
                    auto init =
                        builder.create<mlir_ts::ConstantOp>(location, typeDescrType, builder.getIntegerAttr(typeDescrType, 0));
                    return std::make_tuple(init.getType(), init, TypeProvided::Yes);
                },
                genContext);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeBitmap(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                               const GenContext &genContext)
    {
        // no need to generate
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        MLIRCodeLogic mcl(builder, compileOptions);

        // register global
        auto name = TYPE_BITMAP_NAME;
        auto fullClassStaticFieldName = getTypeBitmapMethodName(newClassPtr);

        // target-width: the bitmap method returns the Boehm GC_descr built below (= GC_word, the
        // integer the size of `void *`), and must agree with the descriptor global's type.
        auto funcType = getFunctionType({}, builder.getIntegerType(compileOptions.sizeBits()), false);

        mlirGenFunctionBody(
            location, name, fullClassStaticFieldName, funcType,
            [&](mlir::Location location, const GenContext &genContext) {
                auto bitmapValueType = mth.getTypeBitmapValueType();

                auto nullOp = builder.create<mlir_ts::NullOp>(location, getNullType());
                CAST_A_NULLCHECK(classNull, location, newClassPtr->classType, nullOp, genContext, true);

                auto sizeOfStoreElement =
                    builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), mth.getTypeBitmapValueType());

                auto _8Value = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                                   builder.getIntegerAttr(mth.getIndexType(), 8));
                auto sizeOfStoreElementInBits = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::AsteriskToken),
                    sizeOfStoreElement, _8Value);

                // calc bitmap size
                auto sizeOfType =
                    builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), newClassPtr->classType.getStorageType());

                // calc count of store elements of type size
                auto sizeOfTypeInBitmapTypes = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken), sizeOfType,
                    sizeOfStoreElement);

                // size alligned by size of bits
                auto sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::PlusToken),
                    sizeOfTypeInBitmapTypes, sizeOfStoreElementInBits);

                auto _1I64Value = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                                      builder.getIntegerAttr(mth.getIndexType(), 1));

                sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::MinusToken),
                    sizeOfTypeAligned, _1I64Value);

                sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken),
                    sizeOfTypeAligned, sizeOfStoreElementInBits);

                // allocate in stack
                auto arrayValue = builder.create<mlir_ts::AllocaOp>(location, mlir_ts::RefType::get(bitmapValueType),
                                                                    sizeOfTypeAligned);

                // property ref
                auto count = newClassPtr->fieldsCount();
                for (auto index = 0; (unsigned)index < count; index++)
                {
                    auto fieldInfo = newClassPtr->fieldInfoByIndex(index);
                    // skip virrual table for speed adv.
                    if (index == 0 && isa<mlir_ts::OpaqueType>(fieldInfo.type))
                    {
                        continue;
                    }

                    if (mth.isValueType(fieldInfo.type))
                    {
                        continue;
                    }

                    auto fieldValue = mlirGenPropertyAccessExpression(location, classNull, fieldInfo.id, genContext);
                    assert(fieldValue);
                    auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);

                    // cast to int64
                    CAST_A(fieldAddrAsInt, location, mth.getIndexType(), fieldRef, genContext);

                    // calc index
                    auto calcIndex = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken),
                        fieldAddrAsInt, sizeOfStoreElement);

                    auto elemRef = builder.create<mlir_ts::PointerOffsetRefOp>(
                        location, mlir_ts::RefType::get(bitmapValueType), arrayValue, calcIndex);

                    // calc bit
                    auto indexModIndex = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::PercentToken),
                        calcIndex, sizeOfStoreElementInBits);

                    auto indexMod = builder.create<mlir_ts::CastOp>(location, bitmapValueType, indexModIndex);

                    auto _1Value = builder.create<mlir_ts::ConstantOp>(location, bitmapValueType,
                                                                       builder.getIntegerAttr(bitmapValueType, 1));

                    // 1 << index_mod
                    auto bitValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, bitmapValueType,
                        builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanGreaterThanToken), _1Value, indexMod);

                    // load val
                    auto val = builder.create<mlir_ts::LoadOp>(location, bitmapValueType, elemRef);

                    // apply or
                    auto valWithBit = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, bitmapValueType, builder.getI32IntegerAttr((int)SyntaxKind::BarToken), val, bitValue);

                    // save value
                    auto saveToElement = builder.create<mlir_ts::StoreOp>(location, valWithBit, elemRef);
                }

                // target-width: GC_make_descriptor's result is a GC_descr (= GC_word, the integer the
                // size of `void *`) - see the lowering in LowerToLLVM.cpp GCMakeDescriptorOpLowering.
                auto typeDescr = builder.create<mlir_ts::GCMakeDescriptorOp>(location, builder.getIntegerType(compileOptions.sizeBits()), arrayValue,
                                                                             sizeOfTypeInBitmapTypes);

                auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
                builder.create<mlir_ts::ReturnValOp>(location, typeDescr, retVarInfo.first);
                return ValueOrLogicalResult(mlir::success());
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassInstanceOfMethod(ClassLikeDeclaration classDeclarationAST,
                                                     ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        // if (newClassPtr->getHasVirtualTable())
        {
            if (newClassPtr->hasRTTI)
            {
                return mlir::success();
            }

            newClassPtr->hasRTTI = true;

            NodeFactory nf(NodeFactoryFlags::None);

            ts::Block body = undefined;
            if (!newClassPtr->isDeclaration)
            {
                NodeArray<Statement> statements;

                /*
                if (!newClassPtr->baseClasses.empty())
                {
                    auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                    auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                    statements.push_back(nf.createExpressionStatement(callSuper));
                }
                */

                // access .rtti via this (as virtual method)
                // auto cmpRttiToParam = nf.createBinaryExpression(
                //     nf.createIdentifier(LINSTANCEOF_PARAM_NAME), nf.createToken(SyntaxKind::EqualsEqualsToken),
                //     nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword),
                //                                       nf.createIdentifier(S(RTTI_NAME))));

                // access .rtti via static field
                auto fullClassStaticFieldName = concat(newClassPtr->fullName, RTTI_NAME);

                auto cmpRttiToParam = nf.createBinaryExpression(
                     nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)), nf.createToken(SyntaxKind::EqualsEqualsToken),
                     nf.createIdentifier(convertUTF8toWide(std::string(fullClassStaticFieldName))));

                auto cmpLogic = cmpRttiToParam;

                if (!newClassPtr->baseClasses.empty())
                {
                    NodeArray<Expression> argumentsArray;
                    argumentsArray.push_back(nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)));
                    cmpLogic =
                        nf.createBinaryExpression(cmpRttiToParam, nf.createToken(SyntaxKind::BarBarToken),
                                                  nf.createCallExpression(nf.createPropertyAccessExpression(
                                                                              nf.createToken(SyntaxKind::SuperKeyword),
                                                                              nf.createIdentifier(S(INSTANCEOF_NAME))),
                                                                          undefined, argumentsArray));
                }

                auto returnStat = nf.createReturnStatement(cmpLogic);
                statements.push_back(returnStat);

                body = nf.createBlock(statements, false);
            }

            NodeArray<ParameterDeclaration> parameters;
            parameters.push_back(nf.createParameterDeclaration(undefined, undefined,
                                                               nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)), undefined,
                                                               nf.createToken(SyntaxKind::StringKeyword), undefined));

            ModifiersArray modifiers;
            if (newClassPtr->isExport || newClassPtr->isImport)
            {
                modifiers.push_back(nf.createToken(SyntaxKind::PublicKeyword));
            }

            auto instanceOfMethod = nf.createMethodDeclaration(
                modifiers, undefined, nf.createIdentifier(S(INSTANCEOF_NAME)), undefined, undefined,
                parameters, nf.createToken(SyntaxKind::BooleanKeyword), body);

            instanceOfMethod->internalFlags |= InternalFlags::ForceVirtual;
            // TODO: you adding new member to the same DOM(parse) instance but it is used for 2 instances of generic
            // type ERROR: do not change members!!!!

            // INFO: .instanceOf must be first element in VTable for Cast Any
            for (auto member : newClassPtr->extraMembers)
            {
                assert(member == SyntaxKind::Constructor);
            }

            newClassPtr->extraMembers.push_back(instanceOfMethod);
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCreateInterfaceVTableForClass(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              const GenContext &genContext)
    {
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(
                mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr, newInterfacePtr, genContext)))
        {
            return resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        }

        return mlir::failure();
    }

    std::pair<mlir::LogicalResult, MethodInfo *> MLIRGenImpl::findErasedGenericMethod(mlir::Location location,
                                                                                    ClassInfo::TypePtr classInfo,
                                                                                    mlir::StringRef name,
                                                                                    const GenContext &genContext)
    {
        // made before, and in this module: a discovery pass makes its own in a module it then drops
        for (auto &erased : classInfo->erasedGenericMethods)
        {
            if (erased.name == name && theModule.lookupSymbol(erased.funcName))
            {
                return {mlir::success(), &erased};
            }
        }

        auto index = classInfo->getGenericMethodIndex(name);
        if (index < 0)
        {
            // the nearest base class's
            for (auto &base : classInfo->baseClasses)
            {
                auto [result, found] = findErasedGenericMethod(location, base, name, genContext);
                if (mlir::failed(result) || found)
                {
                    return {result, found};
                }
            }

            return {mlir::success(), nullptr};
        }

        auto &genericMethodInfo = classInfo->staticGenericMethods[index];
        if (genericMethodInfo.isStatic)
        {
            return {mlir::success(), nullptr};
        }

        auto [result, funcType, funcName] =
            instantiateErasedGenericFunction(location, genericMethodInfo.funcProto->getName(), genContext);
        if (mlir::failed(result))
        {
            emitError(location) << "can't instantiate generic method '" << name << "' of class '" << classInfo->fullName
                                << "' with 'any' for an interface";
            return {mlir::failure(), nullptr};
        }

        for (auto &erased : classInfo->erasedGenericMethods)
        {
            if (erased.name == name)
            {
                erased.funcName = funcName;
                erased.funcType = funcType;
                return {mlir::success(), &erased};
            }
        }

        MethodInfo methodInfo{};
        methodInfo.name = name.str();
        methodInfo.funcName = funcName;
        methodInfo.funcType = funcType;
        methodInfo.virtualIndex = -1;
        methodInfo.accessLevel = genericMethodInfo.accessLevel;
        classInfo->erasedGenericMethods.push_back(methodInfo);
        return {mlir::success(), &classInfo->erasedGenericMethods.back()};
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location,
                                                                       ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr,
                                                                       const GenContext &genContext)
    {

        MLIRCodeLogic mcl(builder, compileOptions);

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        // TODO: ...
        auto classStorageType = mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType());

        llvm::SmallVector<VirtualMethodOrFieldInfo, 1> virtualTable;
        // the fields stored as optional<T> for an interface's `v?: T`: their slots are tagged
        llvm::SmallVector<mlir::Attribute> optionalStorageFields;
        // the interface's function-typed fields (`fn: () => int`) the class implements with a method
        // (`fn(): int`): their slots point at a cell holding the method (see INTERFACE_METHOD_CELL_TAG)
        llvm::SmallVector<std::pair<mlir::Attribute, MethodInfo>, 1> methodsForFields;
        // of those, the ones filled by an optional class field (`fn?: () => int`): their cells carry the
        // offset of the field's flag, so an access reads the instance's presence
        llvm::SmallVector<mlir::Attribute> optionalFieldsForCells;
        // and the ones filled by a class field stored as the interface field's function made a hybrid: their
        // cells carry the field's offset, so a write through the interface (`h.f = g`) stores to the instance
        llvm::SmallVector<mlir::Attribute> writableFieldsForCells;
        // the interface's fields (not function-typed) the class implements with accessors (`get prop()`): their
        // slots name a cell holding the getter and the setter (see INTERFACE_ACCESSOR_CELL_TAG)
        struct AccessorsForField
        {
            mlir::Attribute id;
            MethodInfo get;
            MethodInfo set;
        };
        llvm::SmallVector<AccessorsForField, 1> accessorsForFields;
        // the class's accessor of a name, or its nearest base class's
        std::function<AccessorInfo *(ClassInfo::TypePtr, mlir::StringRef)> findAccessor =
            [&](ClassInfo::TypePtr classInfo, mlir::StringRef name) -> AccessorInfo * {
            auto index = classInfo->getAccessorIndex(name);
            if (index >= 0)
            {
                return classInfo->accessors[index].isStatic ? nullptr : &classInfo->accessors[index];
            }

            for (auto &base : classInfo->baseClasses)
            {
                if (auto *found = findAccessor(base, name))
                {
                    return found;
                }
            }

            return nullptr;
        };
        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> std::pair<mlir_ts::FieldInfo, mlir::LogicalResult> {
                auto found = false;
                auto foundField = newClassPtr->findField(id, found);
                auto accessorName = dyn_cast<mlir::StringAttr>(id);
                auto *foundAccessor = !found && accessorName && !isa<mlir_ts::FunctionType>(fieldType)
                    ? findAccessor(newClassPtr, accessorName.getValue()) : nullptr;
                if (foundAccessor)
                {
                    auto getType = foundAccessor->get ? foundAccessor->get.funcType.getResult(0) : mlir::Type();
                    auto setType = foundAccessor->set ? foundAccessor->set.funcType.getInput(1) : mlir::Type();
                    if (isConditional || (getType && getType != fieldType) || (setType && setType != fieldType))
                    {
                        emitError(location) << "accessor type not matching for " << id << ":" << to_print(fieldType)
                                            << " for interface '" << newInterfacePtr->fullName << "' in class '"
                                            << newClassPtr->fullName << "'"
                                            << (isConditional ? " (an optional field can't be filled by an accessor)" : "");
                        return {emptyFieldInfo, mlir::failure()};
                    }

                    if (llvm::none_of(accessorsForFields, [&](auto &accessorsForField) { return accessorsForField.id == id; }))
                    {
                        auto methodInfoOf = [&](FunctionEntry funcEntry) {
                            MethodInfo methodInfo{};
                            if (funcEntry)
                            {
                                methodInfo.name = funcEntry.name;
                                methodInfo.funcName = funcEntry.name;
                                methodInfo.dllName = funcEntry.dllName;
                                methodInfo.funcType = funcEntry.funcType;
                                methodInfo.virtualIndex = -1;
                                methodInfo.accessLevel = mlir_ts::AccessLevel::Public;
                            }

                            return methodInfo;
                        };

                        accessorsForFields.push_back({id, methodInfoOf(foundAccessor->get), methodInfoOf(foundAccessor->set)});
                    }

                    return {mlir_ts::FieldInfo{id, fieldType, false, mlir_ts::AccessLevel::Public}, mlir::success()};
                }

                if (!found && isa<mlir_ts::FunctionType>(fieldType))
                {
                    auto fieldName = dyn_cast<mlir::StringAttr>(id);
                    auto *foundMethodPtr = fieldName ? newClassPtr->findMethod(fieldName.getValue()) : nullptr;
                    if (foundMethodPtr && !foundMethodPtr->isStatic)
                    {
                        // the method's first input is its `this`, the field's type has none
                        auto match = mth.TestFunctionTypesMatchWithObjectMethods(
                            location, foundMethodPtr->funcType, mlir::cast<mlir_ts::FunctionType>(fieldType), 1, 0);
                        if (match.result != MatchResultType::Match)
                        {
                            emitError(location) << "method signature not matching '" << fieldName.getValue() << ":"
                                                << to_print(fieldType) << "' for interface '" << newInterfacePtr->fullName
                                                << "' in class '" << newClassPtr->fullName << "'."
                                                << " Found method: " << fieldName.getValue() << ":"
                                                << to_print(foundMethodPtr->funcType);
                            return {emptyFieldInfo, mlir::failure()};
                        }

                        // a field reached twice through a diamond `extends` has two slots and one cell
                        if (llvm::none_of(methodsForFields, [&](auto &methodForField) { return methodForField.first == id; }))
                        {
                            methodsForFields.push_back({id, *foundMethodPtr});
                        }

                        return {mlir_ts::FieldInfo{id, fieldType, false, mlir_ts::AccessLevel::Public}, mlir::success()};
                    }
                }

                // a class field holding a function (`fn: () => int = ...`) stores a hybrid function -
                // {function, `this` or null} - or, unannotated (`fn = () => 7`), a plain one: neither can be handed out
                // as a function taking the object as its `this`. A thunk calling `this.fn(...)` fills the
                // slot as a method does; it reads the field at each call, so a field assigned later is seen.
                // An optional field (`fn?: () => int`) fills only an optional one (`fn?: () => int`), and its cell
                // carries the offset of the field's flag: the cell is shared, the flag is the instance's.
                auto optionalField = found && isa<mlir_ts::OptionalType>(foundField.type);
                if (found && isa<mlir_ts::FunctionType>(fieldType) && fieldType != foundField.type
                    && (!optionalField || isConditional) && mth.isAnyFunctionType(foundField.type))
                {
                    auto fieldName = dyn_cast<mlir::StringAttr>(id);
                    auto match = mth.TestFunctionTypesMatchWithObjectMethods(location, foundField.type, fieldType);
                    if (!fieldName || match.result != MatchResultType::Match)
                    {
                        emitError(location) << "field type not matching for " << id << ":" << to_print(fieldType)
                                            << " for interface '" << newInterfacePtr->fullName << "' in class '"
                                            << newClassPtr->fullName << "'. Found field: " << id << ":"
                                            << to_print(foundField.type);
                        return {emptyFieldInfo, mlir::failure()};
                    }

                    // a field reached twice through a diamond `extends` has two slots and one cell
                    if (llvm::none_of(methodsForFields, [&](auto &methodForField) { return methodForField.first == id; }))
                    {
                        auto funcType = mlir::cast<mlir_ts::FunctionType>(fieldType);
                        auto thunkName = concat(interfaceVTableNameForClass(newClassPtr, newInterfacePtr),
                                                fieldName.getValue(), "thunk");
                        if (mlir::failed(generateSynthMethodToCallField(location, newClassPtr->classType, fieldName.getValue(),
                                                                         thunkName, funcType, genContext)))
                        {
                            return {emptyFieldInfo, mlir::failure()};
                        }

                        MethodInfo thunkInfo{};
                        thunkInfo.name = thunkName.str();
                        thunkInfo.funcType = funcType;
                        thunkInfo.funcName = thunkName.str();
                        thunkInfo.virtualIndex = -1;
                        thunkInfo.accessLevel = mlir_ts::AccessLevel::Public;
                        methodsForFields.push_back({id, thunkInfo});
                        if (optionalField)
                        {
                            optionalFieldsForCells.push_back(id);
                        }
                        else if (foundField.type == mth.getInterfaceFieldFunctionAsHybrid(funcType))
                        {
                            writableFieldsForCells.push_back(id);
                        }
                    }

                    return {mlir_ts::FieldInfo{id, fieldType, false, mlir_ts::AccessLevel::Public}, mlir::success()};
                }

                // a class's `v?: T` is stored as optional<T>; the interface's `v?: T` reads it by its flag
                // (a tagged slot, see isStoredAsOptional below)
                auto optionalStorage = found && isConditional && isa<mlir_ts::OptionalType>(foundField.type)
                    && mlir::cast<mlir_ts::OptionalType>(foundField.type).getElementType() == fieldType;
                if (!found || (fieldType != foundField.type && !optionalStorage))
                {
                    if (!found && !isConditional || found)
                    {
                        if (auto valueType = found ? MLIRTypeHelper::getPlainValueForOptionalMember(fieldType, foundField.type) : mlir_ts::ValueType())
                        {
                            MLIRTypeHelper::emitPlainValueForOptionalMemberError(location, id, valueType, newInterfacePtr->fullName);
                            return {emptyFieldInfo, mlir::failure()};
                        }

                        emitError(location)
                            << "field type not matching for " << id << " for interface '" << newInterfacePtr->fullName
                            << "' in class '" << newClassPtr->fullName << "'";

                        return {emptyFieldInfo, mlir::failure()};
                    }

                    return {emptyFieldInfo, mlir::success()};
                }

                if (optionalStorage)
                {
                    optionalStorageFields.push_back(id);
                }

                return {foundField, mlir::success()};
            },
            [&](std::string name, mlir_ts::FunctionType funcType, bool isConditional, int interfacePosIndex) -> std::pair<MethodInfo &, mlir::LogicalResult> {
                auto foundMethodPtr = newClassPtr->findMethod(name);
                // a generic method fills the interface's slot for it erased to `any`
                if (!foundMethodPtr)
                {
                    auto [result, erasedMethodPtr] = findErasedGenericMethod(location, newClassPtr, name, genContext);
                    if (mlir::failed(result))
                    {
                        return {emptyMethod, mlir::failure()};
                    }

                    foundMethodPtr = erasedMethodPtr;
                }

                if (!foundMethodPtr)
                {
                    // TODO: generate method wrapper for calling new/ctor method
                    if (name == NEW_CTOR_METHOD_NAME)
                    {
                        // TODO: generate method                        
                        foundMethodPtr = generateSynthMethodToCallNewCtor(
                            location, newClassPtr, newInterfacePtr, funcType, interfacePosIndex, genContext);
                    }

                    if (!foundMethodPtr)
                    {
                        if (!isConditional)
                        {
                            emitError(location)
                                << "can't find method '" << name << "' for interface '" << newInterfacePtr->fullName
                                << "' in class '" << newClassPtr->fullName << "'";

                            return {emptyMethod, mlir::failure()};
                        }

                        return {emptyMethod, mlir::success()};
                    }
                }

                auto foundMethodFunctionType = foundMethodPtr->funcType;

                auto result = mth.TestFunctionTypesMatch(funcType, foundMethodFunctionType, 1);
                if (result.result != MatchResultType::Match)
                {
                    emitError(location) << "method signature not matching '" << name << ":" << to_print(funcType)
                                        << "' for interface '" << newInterfacePtr->fullName << "' in class '"
                                        << newClassPtr->fullName << "'."
                                        << " Found method: " << name << ":" << to_print(foundMethodFunctionType);
                    return {emptyMethod, mlir::failure()};
                }

                return {*foundMethodPtr, mlir::success()};
            });

        if (mlir::failed(result))
        {
            return result;
        }

        // the header (INTERFACE_VTABLE_HEADER_SLOTS), ahead of the interface's members: the class's
        // type tag, through which an interface value over an object of it is released and retained,
        // and its own `.instanceOf`, so an interface value can be asked what class it holds. A class
        // with no `.instanceOf` (no vtable of its own) is an instance of nothing it can answer for.
        if (auto *instanceOf = newClassPtr->findMethod(INSTANCEOF_NAME))
        {
            virtualTable.insert(virtualTable.begin(), VirtualMethodOrFieldInfo(*instanceOf));
        }
        else
        {
            auto noInstanceOf = mlirGenNoInstanceOfFunction(location);
            virtualTable.insert(virtualTable.begin(), VirtualMethodOrFieldInfo(noInstanceOf));
        }

        virtualTable.insert(virtualTable.begin(), interfaceVTableTypeSlot());

        // a method implementing this interface can itself be owned by a dynamically imported
        // class (not just a base of it) - such a slot can't be a link-time constant SymbolRefOp
        // and must be resolved at runtime instead (see the dynamic-import handling in the method
        // branch below, mirroring mlirGenClassVirtualTableDefinition's existing treatment of its
        // own (non-interface) vtable slots).
        std::function<ClassInfo::TypePtr(ClassInfo::TypePtr, mlir::StringRef)> findMethodOwner =
            [&](ClassInfo::TypePtr cls, mlir::StringRef funcName) -> ClassInfo::TypePtr {
            if (llvm::any_of(cls->methods, [&](auto &m) { return m.funcName == funcName; }))
            {
                return cls;
            }

            for (auto &base : cls->baseClasses)
            {
                if (auto owner = findMethodOwner(base, funcName))
                {
                    return owner;
                }
            }

            return ClassInfo::TypePtr();
        };

        auto needsRuntimeSymbolResolution = llvm::any_of(virtualTable, [&](auto &methodOrField) {
            if (methodOrField.isMissing || methodOrField.isField)
            {
                return false;
            }

            auto owner = findMethodOwner(newClassPtr, methodOrField.methodInfo.funcName);
            return owner && owner->isDynamicImport;
        });

        // the symbol of a method implementing an interface slot - resolved at runtime when the method is owned
        // by a dynamically imported class (`location` is then UnknownLoc, see needsRuntimeSymbolResolution)
        auto methodSymbolValue = [&](mlir::Location location, MethodInfo &methodInfo, const GenContext &genContext) -> mlir::Value {
            auto owner = findMethodOwner(newClassPtr, methodInfo.funcName);
            if (owner && owner->isDynamicImport)
            {
                // `location` is already UnknownLoc here (see
                // needsRuntimeSymbolResolution above, computed from this same
                // owner->isDynamicImport condition) - the resulting
                // SearchForAddressOfSymbolOp forces this whole global's
                // initializer into a synthesized __cctor function via
                // GlobalOpLowering, and a "live" (main-scoped) location on any op
                // living there would attach the wrong DISubprogram to it.
                auto symbolNameValue = V(mlirGenStringValue(location,
                    methodInfo.dllName.empty() ? methodInfo.funcName : methodInfo.dllName, true));
                auto referenceToSymbolOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                    location, getOpaqueType(), symbolNameValue);
                auto castResult = cast(location, methodInfo.funcType, referenceToSymbolOpaque, genContext);
                if (castResult.failed_or_no_value())
                {
                    return mlir::Value();
                }

                return V(castResult);
            }

            return builder.create<mlir_ts::SymbolRefOp>(
                location, methodInfo.funcType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), methodInfo.funcName));
        };

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);

        // one cell per method filling a function-typed field: the field's slot holds its address
        // (tagged), since the object itself has no field holding the method. The variable keeps the
        // name it is registered with, so the name is allocated like the vtable's own.
        auto methodCellName = [&](mlir::Attribute id) {
            return concat(fullClassInterfaceVTableFieldName, mlir::cast<mlir::StringAttr>(id).getValue(), "method");
        };

        for (auto &[id, methodInfo] : methodsForFields)
        {
            auto owner = findMethodOwner(newClassPtr, methodInfo.funcName);
            auto cellType = registerVariable(
                location, methodCellName(id), true, VariableType::Var,
                [&](mlir::Location location, const GenContext &genContext) {
                    if (owner && owner->isDynamicImport)
                    {
                        location = mlir::UnknownLoc::get(builder.getContext());
                    }

                    auto methodValue = methodSymbolValue(location, methodInfo, genContext);
                    if (!methodValue)
                    {
                        return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                    }

                    // the cell is {method, flag offset, field offset}, the method first so the slot is a ref to it.
                    // The flag offset is of the flag of an optional class field (`fn?: () => int`), from `this`; -1
                    // is always present (a method, or a field that is not optional). The field offset is of the
                    // class field a write through the interface stores to; -1 is not written.
                    auto ptrIntType = builder.getIntegerType(compileOptions.sizeBits());
                    auto negative1 = [&]() -> mlir::Value {
                        return builder.create<mlir_ts::ConstantOp>(location, ptrIntType, builder.getIntegerAttr(ptrIntType, -1));
                    };

                    // &(null)->field
                    auto fieldRefOf = [&]() -> mlir::Value {
                        MLIRCodeLogic mcl(builder, compileOptions);
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        auto classNull = cast(location, newClassPtr->classType, nullObj, genContext, true);
                        auto fieldValue = mlirGenPropertyAccessExpression(location, classNull, id, genContext);
                        auto fieldRef = fieldValue ? mcl.GetReferenceFromValue(location, fieldValue) : mlir::Value();
                        if (!fieldRef)
                        {
                            emitError(location) << "can't find reference for field: " << id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                        }

                        return fieldRef;
                    };

                    mlir::Value flagOffsetValue;
                    if (llvm::is_contained(optionalFieldsForCells, id))
                    {
                        // stored as optional<T> - {T, flag}, the flag right after T
                        auto fieldRef = fieldRefOf();
                        if (!fieldRef)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto optionalType = mlir::cast<mlir_ts::OptionalType>(
                            mlir::cast<mlir_ts::RefType>(fieldRef.getType()).getElementType());
                        auto fieldOffset = builder.create<mlir_ts::CastOp>(location, ptrIntType, fieldRef);
                        auto valueSize = builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(),
                                                                           optionalType.getElementType());
                        auto valueSizeInt = builder.create<mlir_ts::CastOp>(location, ptrIntType, valueSize);
                        flagOffsetValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                            location, ptrIntType, builder.getI32IntegerAttr((int)SyntaxKind::PlusToken), fieldOffset,
                            valueSizeInt);
                    }
                    else
                    {
                        flagOffsetValue = negative1();
                    }

                    mlir::Value fieldOffsetValue;
                    if (llvm::is_contained(writableFieldsForCells, id))
                    {
                        auto fieldRef = fieldRefOf();
                        if (!fieldRef)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        fieldOffsetValue = builder.create<mlir_ts::CastOp>(location, ptrIntType, fieldRef);
                    }
                    else
                    {
                        fieldOffsetValue = negative1();
                    }

                    mlir::SmallVector<mlir_ts::FieldInfo> cellFields{
                        {MLIRHelper::TupleFieldName("method", builder.getContext()), methodInfo.funcType, false,
                         mlir_ts::AccessLevel::Public},
                        {MLIRHelper::TupleFieldName("flagOffset", builder.getContext()), ptrIntType, false,
                         mlir_ts::AccessLevel::Public},
                        {MLIRHelper::TupleFieldName("fieldOffset", builder.getContext()), ptrIntType, false,
                         mlir_ts::AccessLevel::Public}};
                    auto cellTupleType = getTupleType(cellFields);
                    mlir::Value cellValue = builder.create<mlir_ts::UndefOp>(location, cellTupleType);
                    cellValue = builder.create<mlir_ts::InsertPropertyOp>(location, cellTupleType, methodValue, cellValue,
                                                                          MLIRHelper::getStructIndex(builder, 0));
                    cellValue = builder.create<mlir_ts::InsertPropertyOp>(location, cellTupleType, flagOffsetValue,
                                                                          cellValue, MLIRHelper::getStructIndex(builder, 1));
                    cellValue = builder.create<mlir_ts::InsertPropertyOp>(location, cellTupleType, fieldOffsetValue,
                                                                          cellValue, MLIRHelper::getStructIndex(builder, 2));
                    return TypeValueInitType{cellTupleType, cellValue, TypeProvided::Yes};
                },
                genContext);
            if (!cellType)
            {
                return mlir::failure();
            }
        }

        // one cell per field filled by accessors: {getter, setter}, either null when the class has none
        auto accessorCellName = [&](mlir::Attribute id) {
            return concat(fullClassInterfaceVTableFieldName, mlir::cast<mlir::StringAttr>(id).getValue(), "accessor");
        };

        for (auto &accessorsForField : accessorsForFields)
        {
            auto cellType = registerVariable(
                location, accessorCellName(accessorsForField.id), true, VariableType::Var,
                [&](mlir::Location location, const GenContext &genContext) {
                    auto isOwnedByDynamicImport = [&](MethodInfo &methodInfo) {
                        auto owner = methodInfo.funcType ? findMethodOwner(newClassPtr, methodInfo.funcName) : ClassInfo::TypePtr();
                        return owner && owner->isDynamicImport;
                    };
                    if (isOwnedByDynamicImport(accessorsForField.get) || isOwnedByDynamicImport(accessorsForField.set))
                    {
                        location = mlir::UnknownLoc::get(builder.getContext());
                    }

                    auto accessorValue = [&](MethodInfo &methodInfo) -> mlir::Value {
                        if (!methodInfo.funcType)
                        {
                            auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
                            return cast(location, getOpaqueType(), nullValue, genContext, true);
                        }

                        auto methodValue = methodSymbolValue(location, methodInfo, genContext);
                        return methodValue ? builder.create<mlir_ts::CastOp>(location, getOpaqueType(), methodValue) : mlir::Value();
                    };

                    auto getterValue = accessorValue(accessorsForField.get);
                    auto setterValue = accessorValue(accessorsForField.set);
                    if (!getterValue || !setterValue)
                    {
                        return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                    }

                    mlir::SmallVector<mlir_ts::FieldInfo> cellFields{
                        {MLIRHelper::TupleFieldName("get", builder.getContext()), getOpaqueType(), false,
                         mlir_ts::AccessLevel::Public},
                        {MLIRHelper::TupleFieldName("set", builder.getContext()), getOpaqueType(), false,
                         mlir_ts::AccessLevel::Public}};
                    auto cellTupleType = getTupleType(cellFields);
                    mlir::Value cellValue = builder.create<mlir_ts::UndefOp>(location, cellTupleType);
                    cellValue = builder.create<mlir_ts::InsertPropertyOp>(location, cellTupleType, getterValue, cellValue,
                                                                          MLIRHelper::getStructIndex(builder, 0));
                    cellValue = builder.create<mlir_ts::InsertPropertyOp>(location, cellTupleType, setterValue, cellValue,
                                                                          MLIRHelper::getStructIndex(builder, 1));
                    return TypeValueInitType{cellTupleType, cellValue, TypeProvided::Yes};
                },
                genContext);
            if (!cellType)
            {
                return mlir::failure();
            }
        }

        auto registeredType = registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) {
                // build vtable from names of methods

                // this global's initializer can be rebuilt lazily from a call site deep inside
                // an unrelated function (a cast expression in main(), via
                // mlirGenCreateInterfaceVTableForClass); `location` still carries that call
                // site's debug scope. Once any slot needs runtime symbol resolution,
                // GlobalOpLowering promotes the WHOLE initializer into a synthesized __cctor
                // function with its own DISubprogram (see the comment there) - so EVERY op
                // built in this callback (not just the one slot that triggered the promotion)
                // ends up living in __cctor, and must not carry a `!dbg` pointing at the
                // calling site's (wrong) subprogram. This is compiler-synthesized glue, not
                // user source worth stepping through, so drop location info entirely rather
                // than try to keep it consistent per-op.
                if (needsRuntimeSymbolResolution)
                {
                    location = mlir::UnknownLoc::get(builder.getContext());
                }

                MLIRCodeLogic mcl(builder, compileOptions);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (isInterfaceVTableTypeSlot(methodOrField))
                    {
                        auto typeTag = mlirGenInterfaceVTableTypeTag(location, newClassPtr->classType);
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, typeTag, vtableValue, MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else if (methodOrField.isMissing)
                    {
                        // a genuinely-absent optional (`?`) interface field/method the class
                        // doesn't implement - mirrors the object-literal handling in
                        // mlirGenObjectVirtualTableDefinitionForInterface
                        // (MLIRGenInterfaces.cpp): querying the class for a field/method it
                        // doesn't have (via mlirGenPropertyAccessExpression, below) crashes
                        // internally instead of failing gracefully, so this case must be
                        // detected and handled up front rather than falling into the "present"
                        // branches - use the same -1 sentinel placeholder, cast to the slot's
                        // ref/func-pointer type.
                        // target-width: a -1 pointer sentinel, cast straight to a ref/func-pointer slot
                        // below. The lowering that tests it (InterfaceSymbolRefOpLowering in
                        // LowerToLLVM.cpp) builds its -1 at compileOptions.sizeBits(), so this must match.
                        auto ptrIntType = builder.getIntegerType(compileOptions.sizeBits());
                        auto negative1 = builder.create<mlir_ts::ConstantOp>(location, ptrIntType,
                                                                             builder.getIntegerAttr(ptrIntType, -1));
                        auto slotType = methodOrField.isField ? methodOrField.fieldInfo.type : methodOrField.methodInfo.funcType;
                        auto castedPtr = cast(location, mlir_ts::RefType::get(slotType), negative1, genContext);
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, castedPtr, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else if (methodOrField.isField && llvm::any_of(methodsForFields, [&](auto &methodForField) {
                                 return methodForField.first == methodOrField.fieldInfo.id;
                             }))
                    {
                        // a method filling a function-typed field: the slot is the address of its cell plus
                        // INTERFACE_METHOD_CELL_TAG, so an access reads the cell instead of `this` + offset
                        // (InterfaceSymbolRefOpLowering)
                        auto cellRef = resolveFullNameIdentifier(location, methodCellName(methodOrField.fieldInfo.id), true, genContext);
                        if (!cellRef)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto ptrIntType = builder.getIntegerType(compileOptions.sizeBits());
                        auto cellAddress = builder.create<mlir_ts::CastOp>(location, ptrIntType, cellRef);
                        auto tagValue = builder.create<mlir_ts::ConstantOp>(
                            location, ptrIntType, builder.getIntegerAttr(ptrIntType, INTERFACE_METHOD_CELL_TAG));
                        auto taggedValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                            location, ptrIntType, builder.getI32IntegerAttr((int)SyntaxKind::PlusToken), cellAddress,
                            tagValue);
                        auto slotValue = builder.create<mlir_ts::CastOp>(
                            location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), taggedValue);

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, slotValue, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else if (methodOrField.isField && llvm::any_of(accessorsForFields, [&](auto &accessorsForField) {
                                 return accessorsForField.id == methodOrField.fieldInfo.id;
                             }))
                    {
                        // a field filled by accessors: the slot is the address of their cell plus
                        // INTERFACE_ACCESSOR_CELL_TAG, so a load or a store through it calls them
                        auto cellRef = resolveFullNameIdentifier(location, accessorCellName(methodOrField.fieldInfo.id), true, genContext);
                        if (!cellRef)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto ptrIntType = builder.getIntegerType(compileOptions.sizeBits());
                        auto cellAddress = builder.create<mlir_ts::CastOp>(location, ptrIntType, cellRef);
                        auto tagValue = builder.create<mlir_ts::ConstantOp>(
                            location, ptrIntType,
                            builder.getIntegerAttr(ptrIntType, llvm::APInt::getSignMask(compileOptions.sizeBits())));
                        auto taggedValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                            location, ptrIntType, builder.getI32IntegerAttr((int)SyntaxKind::PlusToken), cellAddress, tagValue);
                        auto slotValue = builder.create<mlir_ts::CastOp>(
                            location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), taggedValue);

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, slotValue, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        auto classNull = cast(location, newClassPtr->classType, nullObj, genContext, true);
                        auto fieldValue = mlirGenPropertyAccessExpression(location, classNull,
                                                                          methodOrField.fieldInfo.id, genContext);
                        if (!fieldValue)
                        {
                            emitError(location) << "can't find field (or it is inaccessible): " << methodOrField.fieldInfo.id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);
                        if (!fieldRef)
                        {
                            emitError(location) << "can't find reference for field: " << methodOrField.fieldInfo.id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        // stored as optional<T> - {T, flag}, value first - for an interface's `v?: T`: the slot is
                        // the offset plus INTERFACE_OPTIONAL_STORAGE_TAG, so an access reads the flag
                        // (InterfaceSymbolRefOpLowering)
                        if (llvm::is_contained(optionalStorageFields, methodOrField.fieldInfo.id))
                        {
                            auto ptrIntType = builder.getIntegerType(compileOptions.sizeBits());
                            auto offsetValue = builder.create<mlir_ts::CastOp>(location, ptrIntType, fieldRef);
                            auto tagValue = builder.create<mlir_ts::ConstantOp>(
                                location, ptrIntType,
                                builder.getIntegerAttr(ptrIntType, INTERFACE_OPTIONAL_STORAGE_TAG(compileOptions.sizeBits())));
                            auto taggedValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                                location, ptrIntType, builder.getI32IntegerAttr((int)SyntaxKind::PlusToken), offsetValue,
                                tagValue);
                            fieldRef = builder.create<mlir_ts::CastOp>(location, fieldRef.getType(), taggedValue);
                        }

                        // insert &(null)->field
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, fieldRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else
                    {
                        // the interface-implementing method can itself be owned by a
                        // dynamically imported class (not just a base of it - the class
                        // satisfying the interface may live in another module entirely) - same
                        // "not a link-time constant, resolve at runtime" situation
                        // mlirGenClassVirtualTableDefinition already handles for its own
                        // (non-interface) vtable slots, just not previously mirrored here.
                        auto methodValueRef = methodSymbolValue(location, methodOrField.methodInfo, genContext);
                        if (!methodValueRef)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, methodValueRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }

                    fieldIndex++;
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return registeredType ? mlir::success() : mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassBaseInterfaces(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                   const GenContext &genContext)
    {
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        for (auto &baseClass : newClassPtr->baseClasses)
        {
            for (auto &implement : baseClass->implements)
            {
                if (mlir::failed(mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr,
                                                                                implement.interface, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             HeritageClause heritageClause,
                                                             const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ImplementsKeyword)
        {
            return mlir::success();
        }

        for (auto &implementingType : heritageClause->types)
        {
            auto result = mlirGen(implementingType, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto ifaceType = V(result);
            auto success = false;
            mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    assert(interfaceInfo);
                    if (!newClassPtr->isDeclaration)
                    {
                        success = !failed(mlirGenClassVirtualTableDefinitionForInterface(
                            loc(implementingType), newClassPtr, interfaceInfo, genContext));
                    }
                    else
                    {
                        success = true;
                    }
                })
                .Default([&](auto type) {
                    // same shape as mlirGenClassHeritageClause's `implements` branch above (a
                    // type-alias-resolved TupleType target, not a genuine InterfaceType) - this
                    // function builds the vtable definition for whatever `implements` already
                    // registered, so it hits the same real gap from a different stage.
                    emitError(loc(implementingType)) << "class cannot implement type: " << to_print(type);
                });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassVirtualTableDefinition(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                           const GenContext &genContext)
    {
        // an abstract class has one too, though no instance carries it: a value of the class
        // (`const k: typeof A = A`) reads its static members through it. Its abstract methods'
        // and its interfaces' slots are null, as only a static is read through it.
        if (!newClassPtr->getHasVirtualTable())
        {
            return mlir::success();
        }

        auto nullSlot = [&](mlir::Location location, mlir::Type slotType) -> mlir::Value {
            // through Opaque: a null function is no TypeScript value
            auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
            auto nullOpaque = builder.create<mlir_ts::CastOp>(location, getOpaqueType(), nullValue);
            auto castResult = cast(location, slotType, nullOpaque, genContext);
            return castResult.failed_or_no_value() ? mlir::Value() : V(castResult);
        };

        // TODO: ...
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo, 1> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        // TODO: this is pure hack, add ability to clean up created globals while "dummyRun = true"
        // look into examnple with class declaraion in generic function
        auto fullClassVTableFieldName = concat(newClassPtr->fullName, VTABLE_NAME);
        if (fullNameGlobalsMap.count(fullClassVTableFieldName))
        {
            return mlir::success();
        }

        // register global
        VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
        varClass.isExport = newClassPtr->isExport;
        varClass.isImport = newClassPtr->isImport;
        varClass.isPublic = newClassPtr->isPublic;            
        auto vtableRegisteredType = registerVariable(
            location, fullClassVTableFieldName, true,
            varClass,
            [&](mlir::Location location, const GenContext &genContext) {
                auto virtTuple = getVirtualTableType(virtualTable);
                if (newClassPtr->isDeclaration)
                {
                    return TypeValueInitType{virtTuple, mlir::Value(), TypeProvided::Yes};
                }

                // build vtable from names of methods
                MLIRCodeLogic mcl(builder, compileOptions);
                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto vtRecord : virtualTable)
                {
                    if (vtRecord.isInterfaceVTable && newClassPtr->isAbstract)
                    {
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, nullSlot(location, getOpaqueType()), vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex++));
                    }
                    else if (vtRecord.isInterfaceVTable)
                    {
                        // TODO: write correct full name for vtable
                        auto fullClassInterfaceVTableFieldName =
                            concat(newClassPtr->fullName, vtRecord.methodInfo.name, VTABLE_NAME);
                        auto interfaceVTableValue =
                            resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);

                        if (!interfaceVTableValue)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto interfaceVTableValueAsAny =
                            cast(location, getOpaqueType(), interfaceVTableValue, genContext);

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, interfaceVTableValueAsAny, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex++));
                    }
                    else
                    {
                        // The vtable is extends-recursive, so a derived class's vtable can
                        // contain slots whose symbols (inherited virtual methods, and - with
                        // ADD_STATIC_MEMBERS_TO_VTABLE - inherited static fields like the
                        // RTTI `.rtti`/`.size`) are owned by a base class that lives in a
                        // dynamically imported module. Those cannot be constant SymbolRefOp
                        // references: with no import library, the address is not a link-time
                        // constant. Resolve them at runtime instead (SearchForAddressOfSymbolOp
                        // + cast) - GlobalOpLowering already routes any initializer containing
                        // a SearchForAddressOfSymbolOp through the __cctor global-constructor
                        // path, and the module-load ctor is emitted before all per-symbol
                        // ctors, so the DLL is loaded by the time this resolves.
                        auto isOwnedByDynamicImport = [&](mlir::StringRef symbolName, bool isStaticField) {
                            std::function<ClassInfo::TypePtr(ClassInfo::TypePtr)> findOwner =
                                [&](ClassInfo::TypePtr cls) -> ClassInfo::TypePtr {
                                if (isStaticField
                                        ? llvm::any_of(cls->staticFields, [&](auto &f) { return f.globalVariableName == symbolName; })
                                        : llvm::any_of(cls->methods, [&](auto &m) { return m.funcName == symbolName; }))
                                {
                                    return cls;
                                }

                                for (auto &base : cls->baseClasses)
                                {
                                    if (auto owner = findOwner(base))
                                    {
                                        return owner;
                                    }
                                }

                                return ClassInfo::TypePtr();
                            };

                            auto owner = findOwner(newClassPtr);
                            return owner && owner->isDynamicImport;
                        };

                        mlir::Value methodOrFieldNameRef;
                        mlir::StringRef symbolName;
                        mlir::Type slotType;
                        if (!vtRecord.isStaticField)
                        {
                            if (vtRecord.methodInfo.isAbstract && !newClassPtr->isAbstract)
                            {
                                emitError(location) << "Abstract method '" << vtRecord.methodInfo.name <<  "' is not implemented in '" << newClassPtr->name << "'";
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                            }

                            symbolName = vtRecord.methodInfo.funcName;
                            slotType = vtRecord.methodInfo.funcType;
                        }
                        else
                        {
                            symbolName = vtRecord.staticFieldInfo.globalVariableName;
                            slotType = mlir_ts::RefType::get(vtRecord.staticFieldInfo.type);
                        }

                        // a class implementing C++ methods: a slot it leaves to a method of the
                        // C++ class it extends, which has no body, is null, as a pure virtual's
                        auto isUnimplementedCppMethod = [&]() {
                            if (!newClassPtr->isCppVTable || vtRecord.isStaticField)
                            {
                                return false;
                            }

                            // a slot no method names (@slot skipped it), or one named by a method of
                            // this class without a body
                            if (vtRecord.methodInfo.isCppSlotGap)
                            {
                                return true;
                            }

                            if (cppMethodsWithoutBody.contains(symbolName))
                            {
                                return true;
                            }

                            std::function<ClassInfo::TypePtr(ClassInfo::TypePtr)> findOwner =
                                [&](ClassInfo::TypePtr cls) -> ClassInfo::TypePtr {
                                if (llvm::any_of(cls->methods, [&](auto &m) { return m.funcName == symbolName; }))
                                {
                                    return cls;
                                }

                                for (auto &base : cls->baseClasses)
                                {
                                    if (auto owner = findOwner(base))
                                    {
                                        return owner;
                                    }
                                }

                                return ClassInfo::TypePtr();
                            };

                            auto owner = findOwner(newClassPtr);
                            return owner && owner->isCppVTable && !owner->isCppVTableImplementation;
                        };

                        if (isUnimplementedCppMethod() || (!vtRecord.isStaticField && vtRecord.methodInfo.isAbstract))
                        {
                            methodOrFieldNameRef = nullSlot(location, slotType);
                            if (!methodOrFieldNameRef)
                            {
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                            }
                        }
                        else if (isOwnedByDynamicImport(symbolName, vtRecord.isStaticField))
                        {
                            auto dllSymbolName = !vtRecord.isStaticField && !vtRecord.methodInfo.dllName.empty()
                                ? StringRef(vtRecord.methodInfo.dllName)
                                : symbolName;
                            auto symbolNameValue = V(mlirGenStringValue(location, dllSymbolName.str(), true));
                            auto referenceToSymbolOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                                location, getOpaqueType(), symbolNameValue);
                            auto castResult = cast(location, slotType, referenceToSymbolOpaque, genContext);
                            if (castResult.failed_or_no_value())
                            {
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                            }

                            methodOrFieldNameRef = V(castResult);
                        }
                        else
                        {
                            methodOrFieldNameRef = builder.create<mlir_ts::SymbolRefOp>(
                                location, slotType,
                                mlir::FlatSymbolRefAttr::get(builder.getContext(), symbolName));
                        }

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, methodOrFieldNameRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex++));
                    }
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return (vtableRegisteredType) ? mlir::success() : mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassIndexMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                const GenContext &genContext)
    {
        if (classMember->processed)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n\tALREADY PROCESSED.");
            return mlir::success();
        }

        // TODO:
        auto indexElement = classMember.as<IndexSignatureDeclaration>();

        auto &indexInfos = newClassPtr->indexes;

        auto res = mlirGenFunctionSignaturePrototype(indexElement.as<SignatureDeclaration>(), false, genContext);
        auto funcType = std::get<1>(res);

        LLVM_DEBUG(llvm::dbgs() << "\n\tindex signature: " << funcType << "\n");

        if (std::find_if(
            indexInfos.begin(), 
            indexInfos.end(), 
            [&] (auto& item) { 
                return item.indexSignature == funcType; 
            }) == indexInfos.end())
        {
            indexInfos.push_back({funcType, {}, {}});
        } 

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 int orderWeight,
                                                 const GenContext &genContext)
    {
        if (classMember->processed)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n\tALREADY PROCESSED.");
            return mlir::success();
        }

        ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);
        if (!classMethodMemberInfo.isFunctionLike())
        {
            // process indexer here
            return mlir::success();
        }

        auto location = loc(classMember);

        auto accessLevel = getAccessLevel(classMember);
        auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();

        // an overload signature of a method or a constructor, as for a function: the implementation
        // that follows it is the member
        // (a method of a @cpp class without a body is a C++ method, a class implementing some has
        // methods of both kinds, and none is overloaded)
        auto isAmbientMember = newClassPtr->isDeclaration || newClassPtr->isImport
            || (classMember->flags & NodeFlags::Ambient) == NodeFlags::Ambient
            || (newClassPtr->isCppVTable && classMember == SyntaxKind::MethodDeclaration);
        if (!funcLikeDeclaration->body && !classMethodMemberInfo.isAbstract && !isAmbientMember
            && (classMember == SyntaxKind::MethodDeclaration || classMember == SyntaxKind::Constructor))
        {
            return mlir::success();
        }

        if (mlir::failed(getMethodNameOrPropertyName(
            newClassPtr->isStatic,
            funcLikeDeclaration, 
            classMethodMemberInfo.methodName, 
            classMethodMemberInfo.propertyName, 
            genContext)))
        {
            return mlir::failure();
        }

        assert (!classMethodMemberInfo.methodName.empty());

        // a method of a @cpp class is virtual only as a C++ virtual method (isCppVirtualMethod)
        if (newClassPtr->isCppVTable && !classMethodMemberInfo.isConstructor && !classMethodMemberInfo.isStatic)
        {
            classMethodMemberInfo.cppSlot = std::max(getCppSlot(classMember, genContext), -1);
            classMethodMemberInfo.isVirtual = isCppVirtualMethod(newClassPtr, classMember, genContext);
            classMethodMemberInfo.isForceVirtual = classMethodMemberInfo.isVirtual;
        }

        // update access based on name
        if (StringRef(classMethodMemberInfo.getName()).starts_with("#")) {
            accessLevel = mlir_ts::AccessLevel::Private;
        }

        if (classMethodMemberInfo.isAbstract && !newClassPtr->isAbstract)
        {
            emitError(location) << "Can't use abstract member '" 
                << classMethodMemberInfo.getName()
                << "' in non-abstract class '" << newClassPtr->fullName << "'";
            return mlir::failure();
        }

        classMember->parent = classDeclarationAST;

        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        funcGenContext.thisType = newClassPtr->classType;
        funcGenContext.thisClassType = newClassPtr->classType;
        if (!classMethodMemberInfo.isStatic && !classMethodMemberInfo.isConstructor && !funcLikeDeclaration->type)
        {
            if (auto baseReturnType = overriddenNumericReturnType(newClassPtr, classMethodMemberInfo.methodName))
            {
                funcGenContext.overriddenReturnType = baseReturnType;
                funcGenContext.overriddenReturnFor = funcLikeDeclaration.operator->();
            }
        }
        // `this` of a @value class: in the constructor the slot `new` made, whose fields it assigns
        // (the variable `this` is that slot: mlirGenFunctionParams); in a method or a getter the value
        // itself, a copy. A static member's `this` is the class.
        if (newClassPtr->isValue && !classMethodMemberInfo.isStatic)
        {
            auto valueType = getValueTypeOfClass(newClassPtr);
            funcGenContext.thisType = classMethodMemberInfo.isConstructor
                ? mlir::Type(mlir_ts::RefType::get(valueType))
                : mlir::Type(valueType);
        }

        if (classMethodMemberInfo.isConstructor)
        {
            if (classMethodMemberInfo.isStatic && !genContext.allowPartialResolve)
            {
                createGlobalConstructor(classMember, genContext);
            }

            // adding missing statements
            generateConstructorStatements(classDeclarationAST, classMethodMemberInfo.isStatic, classMember, funcGenContext);
        }

        // process dynamic import
        // TODO: why ".new" is virtual method?
        if (newClassPtr->isDynamicImport 
            && (classMethodMemberInfo.isStatic || classMethodMemberInfo.isConstructor || classMethodMemberInfo.methodName == NEW_METHOD_NAME))
        {
            return mlirGenClassMethodMemberDynamicImport(classMethodMemberInfo, orderWeight, genContext);
        }

        if (classMethodMemberInfo.isExport)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::DllExport;
        }

        if (classMethodMemberInfo.isImport)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::DllImport;
            //MLIRHelper::addDecoratorIfNotPresent(funcLikeDeclaration, DLL_IMPORT);
        }

        if (newClassPtr->isPublic && accessLevel != mlir_ts::AccessLevel::Private)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::IsPublic;
        }

        auto [result, funcOp, funcName, isGeneric] =
            mlirGenFunctionLikeDeclaration(funcLikeDeclaration, funcGenContext);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        if (funcOp)
        {
            classMethodMemberInfo.setFuncOp(funcOp);
            if (classMethodMemberInfo.registerClassMethodMember(loc(funcLikeDeclaration), orderWeight, accessLevel))
            {
                // C++ methods: not a constructor, nor `.new` (static)
                if (!classMethodMemberInfo.isConstructor && !classMethodMemberInfo.isStatic
                    && mlir::failed(mlirGenClassCppVTableHiddenReturn(loc(funcLikeDeclaration), newClassPtr, funcOp,
                                                                      MLIRHelper::hasDecorator(funcLikeDeclaration, CPP_NON_TRIVIAL),
                                                                      static_cast<bool>(funcLikeDeclaration->body),
                                                                      classMethodMemberInfo.isVirtual)))
                {
                    return mlir::failure();
                }

                funcLikeDeclaration->processed = true;
                return mlir::success();
            }

            return mlir::failure();
        }

        return registerGenericClassMethod(classMethodMemberInfo, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticBlockMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 const GenContext &genContext)
    {
        // we need to add all static blocks to it
        if (classMember == SyntaxKind::ClassStaticBlockDeclaration)
        {
            auto classStaticBlock = classMember.as<ClassStaticBlockDeclaration>();

            // create function
            auto location = loc(classStaticBlock);

            auto name = MLIRHelper::getAnonymousName(location, ".csb", "");
            auto fullInitGlobalFuncName = getFullNamespaceName(name);

            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            // create global construct
            auto funcType = getFunctionType({}, {}, false);

            if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, funcType,
                [&](mlir::Location location, const GenContext &genContext) {
                    return mlirGen(classStaticBlock->body, genContext);
                }, genContext)))
            {
                return mlir::failure();
            }

            addGlobalConstructor(location, fullInitGlobalFuncName);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::registerGenericClassMethod(ClassMethodMemberInfo &classMethodMemberInfo, const GenContext &genContext)
    {
        // if funcOp is null, means it is generic
        if (classMethodMemberInfo.funcOp)
        {
            return mlir::success();
        }

        auto funcLikeDeclaration = classMethodMemberInfo.classMember.as<FunctionLikeDeclarationBase>();

        // if it is generic - remove virtual flag
        if (classMethodMemberInfo.isForceVirtual)
        {
            classMethodMemberInfo.isVirtual = false;
        }

        if (classMethodMemberInfo.isStatic || (!classMethodMemberInfo.isAbstract && !classMethodMemberInfo.isVirtual))
        {
            if (classMethodMemberInfo.newClassPtr->getGenericMethodIndex(classMethodMemberInfo.methodName) < 0)
            {
                // a generic method is instantiated for the class a call names, never dispatched: an override
                // would be skipped by every call through a base class's reference
                std::function<bool(ClassInfo::TypePtr)> baseHasGeneric = [&](ClassInfo::TypePtr classInfo) {
                    return llvm::any_of(classInfo->baseClasses, [&](ClassInfo::TypePtr base) {
                        auto index = base->getGenericMethodIndex(classMethodMemberInfo.methodName);
                        return (index >= 0 && !base->staticGenericMethods[index].isStatic) || baseHasGeneric(base);
                    });
                };
                if (!classMethodMemberInfo.isStatic && baseHasGeneric(classMethodMemberInfo.newClassPtr))
                {
                    emitError(loc(classMethodMemberInfo.classMember))
                        << "generic method '" << classMethodMemberInfo.methodName
                        << "' overrides a base class's generic method, which a call through the base class would not "
                           "reach: generic methods are not virtual (declare it in an interface to dispatch it)";
                    return mlir::failure();
                }

                llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
                if (mlir::failed(
                        processTypeParameters(funcLikeDeclaration->typeParameters, typeParameters, genContext)))
                {
                    return mlir::failure();
                }

                // TODO: review it, ignore in case of ArrowFunction,
                auto [result, funcProto] =
                    getFuncArgTypesOfGenericMethod(funcLikeDeclaration, typeParameters, false, genContext);
                if (mlir::failed(result))
                {
                    return mlir::failure();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! registered generic method: " << classMethodMemberInfo.methodName
                                        << ", type: " << funcProto->getFuncType() << "\n";);

                auto &genericMethodInfos = classMethodMemberInfo.newClassPtr->staticGenericMethods;

                // this is generic method
                // the main logic will use Global Generic Functions
                genericMethodInfos.push_back({
                    classMethodMemberInfo.methodName, 
                    funcProto->getFuncType(), 
                    funcProto, 
                    classMethodMemberInfo.isStatic,
                    classMethodMemberInfo.accessLevel});
            }

            return mlir::success();
        }

        emitError(loc(classMethodMemberInfo.classMember)) << "virtual generic methods in class are not allowed";
        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMethodMemberDynamicImport(ClassMethodMemberInfo &classMethodMemberInfo, int orderWeight, const GenContext &genContext)
    {
        auto funcLikeDeclaration = classMethodMemberInfo.classMember.as<FunctionLikeDeclarationBase>();

        auto [funcOp, funcProto, result, isGeneric] =
            mlirGenFunctionPrototype(funcLikeDeclaration, genContext);
        if (mlir::failed(result))
        {
            // in case of ArrowFunction without params and receiver is generic function as well
            return mlir::failure();
        }

        classMethodMemberInfo.setFuncOp(funcOp);

        auto location = loc(funcLikeDeclaration);
        auto dllNameAttr = funcOp->getAttrOfType<mlir::StringAttr>(DLL_NAME);
        if (mlir::succeeded(mlirGenFunctionLikeDeclarationDynamicImport(
            location, funcOp, funcOp.getName(), funcOp.getFunctionType(), dllNameAttr ? dllNameAttr.getValue() : funcOp.getName(), genContext)))
        {
            // no need to generate method in code
            funcLikeDeclaration->processed = true;
            classMethodMemberInfo.registerClassMethodMember(location, orderWeight, classMethodMemberInfo.getAccessLevel());
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::createGlobalConstructor(ClassElement classMember, const GenContext &genContext)
    {
        auto location = loc(classMember);
        auto funcName = getNameOfFunction(classMember, genContext);

        addGlobalConstructor(location, std::get<0>(funcName));

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, bool staticConstructor,
                                                      ClassElement constructorMember, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        auto isClassStatic = hasModifier(classDeclarationAST, SyntaxKind::StaticKeyword);
        for (auto &classMember : classDeclarationAST->members)
        {
            auto isStatic = isClassStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (!propertyDeclaration->initializer)
                {
                    continue;
                }

                if (staticConstructor)
                {
                    auto isConst = isConstValue(propertyDeclaration->initializer, genContext);
                    if (isConst)
                    {
                        continue;
                    }
                }

                auto memberNamePtr = MLIRHelper::getName(propertyDeclaration->name, stringAllocator);
                if (memberNamePtr.empty())
                {
                    // e.g. a computed property name (`[expr] = value;`) reaches here with no
                    // plain-identifier name; tried as a repro but it hits a different, earlier
                    // bug first (an LLVM Casting.h assertion, undocumented/unfixed - see
                    // docs/not-implemented-audit.md), so this specific branch's reachability is
                    // unconfirmed. Fail cleanly rather than crash regardless.
                    emitError(loc(propertyDeclaration), "computed property name is not supported here");
                    return mlir::failure();
                }

                auto _this = nf.createIdentifier(S(THIS_NAME));
                auto _name = nf.createIdentifier(stows(std::string(memberNamePtr)));
                auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                auto _this_name_equal = nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken),
                                                                  propertyDeclaration->initializer);
                auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                // errors in the assignment (the initializer's cast to the field type) are reported at
                // the field, not at the start of the file
                for (Node node : {Node(_this), Node(_name), Node(_this_name), Node(_this_name_equal), Node(expr_statement)})
                {
                    node->pos = propertyDeclaration->pos;
                    node->_end = propertyDeclaration->_end;
                }

                // NOTE: upward mailbox: drained when the constructor body is generated - see GenContext
                const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
            }

            // the parameter properties of this constructor only: another overload's parameters
            // aren't in scope here (`this.data = data` in `constructor(count: int)`)
            if (classMember == SyntaxKind::Constructor && classMember == constructorMember)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
                for (auto &parameter : constructorDeclaration->parameters)
                {
                    auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
                    auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
                    auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

                    if (!(isPublic || isProtected || isPrivate))
                    {
                        continue;
                    }

                    auto propertyNamePtr = MLIRHelper::getName(parameter->name, stringAllocator);
                    if (propertyNamePtr.empty())
                    {
                        // e.g. a parameter property combined with a destructuring pattern
                        // (`constructor(public {x, y}: T) {}`) - real TS rejects this combination
                        // (TS2369) but this compiler doesn't enforce that, so the binding pattern
                        // reaches here with no plain-identifier name. Tried as a repro but it
                        // crashes earlier, in MLIRGenExpressions.cpp's expression dispatcher
                        // (fixed separately this pass), so this specific branch's reachability
                        // is unconfirmed. Fail cleanly rather than crash regardless.
                        emitError(loc(parameter), "a parameter property cannot use a binding pattern");
                        return mlir::failure();
                    }

                    auto _this = nf.createIdentifier(stows(THIS_NAME));
                    auto _name = nf.createIdentifier(stows(std::string(propertyNamePtr)));
                    auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                    auto _this_name_equal =
                        nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), _name);
                    auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                    // NOTE: upward mailbox: drained when the constructor body is generated - see GenContext
                    const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
                }
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenClassValue(mlir::Location location, mlir_ts::ClassType classType,
                                                        const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        if (!classInfo)
        {
            emitError(location) << "Can't find class " << to_print(classType);
            return mlir::failure();
        }

        if (classInfo->isValue)
        {
            emitError(location) << "class '" << classInfo->fullName << "' can't be used as a value yet";
            return mlir::failure();
        }

        // a @cpp class implemented here has its vtable (C++'s layout), `.new` and constructor; one made
        // by C++ only has none of them here
        if (classInfo->isCppVTable && !classInfo->isCppVTableImplementation)
        {
            emitError(location) << "the @cpp class '" << classInfo->fullName
                                << "' can't be used as a value: its objects are made by C++";
            return mlir::failure();
        }

        auto opaqueType = getOpaqueType();
        auto nullOpaque = [&]() -> mlir::Value {
            return builder.create<mlir_ts::CastOp>(location, opaqueType, builder.create<mlir_ts::NullOp>(location, getNullType()));
        };
        // a class of a shared library (-shared) has its functions and its rtti behind pointers the
        // importer resolves at load time (mlirGenClassMethodMemberDynamicImport,
        // mlirGenCustomRTTIDynamicImport): the value is read through them
        auto loadedOpaque = [&](mlir::StringRef name) -> mlir::Value {
            auto value = resolveFullNameIdentifier(location, name, false, genContext);
            if (!value)
            {
                return mlir::Value();
            }

            if (auto refType = dyn_cast<mlir_ts::RefType>(value.getType()))
            {
                value = builder.create<mlir_ts::LoadOp>(location, refType.getElementType(), value);
            }

            return builder.create<mlir_ts::CastOp>(location, opaqueType, value);
        };
        auto functionAddress = [&](MethodInfo *methodInfo) -> mlir::Value {
            if (!methodInfo)
            {
                return nullOpaque();
            }

            // the class that has the method: a class without a constructor of its own has its base's,
            // which may be a shared library's
            std::function<bool(ClassInfo::TypePtr)> ownedByDynamicImport = [&](ClassInfo::TypePtr cls) -> bool {
                if (llvm::any_of(cls->methods, [&](auto &m) { return m.funcName == methodInfo->funcName; }))
                {
                    return cls->isDynamicImport;
                }

                return llvm::any_of(cls->baseClasses, [&](auto &base) { return ownedByDynamicImport(base); });
            };

            if (ownedByDynamicImport(classInfo))
            {
                return loadedOpaque(methodInfo->funcName);
            }

            auto symbolRefOp = builder.create<mlir_ts::SymbolRefOp>(
                location, methodInfo->funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), methodInfo->funcName));
            return builder.create<mlir_ts::CastOp>(location, opaqueType, symbolRefOp);
        };

        // the vtable, for a static read through the value (an abstract class has one for that)
        mlir::Value vtable;
        if (classInfo->isDynamicImport)
        {
            // the importer has no global of it: it is looked up in the library, as a vtable slot a
            // library's class owns is (mlirGenClassVirtualTableDefinition)
            auto vtableName = V(mlirGenStringValue(location, concat(classInfo->fullName, VTABLE_NAME), true));
            vtable = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, opaqueType, vtableName);
        }
        else if (classInfo->getHasVirtualTable())
        {
            // as an instance gets it (mlirGenSetVTableToInstance)
            auto vtableName = concat(classInfo->fullName, VTABLE_NAME);
            if (auto vtableAddress = resolveFullNameIdentifier(location, vtableName, true, genContext))
            {
                vtable = builder.create<mlir_ts::CastOp>(location, opaqueType, vtableAddress);
            }
            else
            {
                vtable = builder.create<mlir_ts::AddressOfOp>(location, opaqueType, vtableName, ::mlir::IntegerAttr());
            }
        }
        else
        {
            vtable = nullOpaque();
        }

        // an abstract class has no `.new`: its value can't make an instance of it
        auto newFunc = functionAddress(classInfo->isAbstract ? nullptr : classInfo->findMethod(NEW_METHOD_NAME));
        auto constructorFunc = functionAddress(classInfo->getHasConstructor() ? classInfo->findMethod(CONSTRUCTOR_NAME) : nullptr);

        // the rtti: every class has one, and no two the same - but a @cpp class, whose identity is its
        // vtable, which each class implementing C++ methods has of its own
        mlir::Value rtti;
        auto rttiName = concat(classInfo->fullName, RTTI_NAME);
        if (classInfo->isCppVTable)
        {
            rtti = vtable;
        }
        else if (classInfo->isDynamicImport)
        {
            rtti = loadedOpaque(rttiName);
        }
        else if (auto rttiAddress = resolveFullNameIdentifier(location, rttiName, true, genContext))
        {
            rtti = builder.create<mlir_ts::CastOp>(location, opaqueType, rttiAddress);
        }
        else
        {
            rtti = builder.create<mlir_ts::AddressOfOp>(location, opaqueType, rttiName, ::mlir::IntegerAttr());
        }

        if (!newFunc || !constructorFunc || !rtti)
        {
            emitError(location) << "class '" << classInfo->fullName << "' of a shared library can't be used as a value: "
                                << "its functions can't be resolved";
            return mlir::failure();
        }

        auto storageType = MLIRTypeHelper::getClassValueStorageType(builder.getContext());
        auto storage = builder.create<mlir_ts::CreateTupleOp>(location, storageType,
                                                              mlir::ValueRange{vtable, newFunc, constructorFunc, rtti});
        return V(builder.create<mlir_ts::CastOp>(location, mlir_ts::ClassValueType::get(classType), storage));
    }

    ValueOrLogicalResult MLIRGenImpl::castToClassValue(mlir::Location location, mlir_ts::ClassValueType classValueType,
                                                       mlir::Value value, const GenContext &genContext)
    {
        auto baseClassType = mlir::cast<mlir_ts::ClassType>(classValueType.getClassType());
        mlir_ts::ClassType classType;
        mlir::Value classValue;
        if (auto classRefOp = value.getDefiningOp<mlir_ts::ClassRefOp>())
        {
            classType = classRefOp.getType();
        }
        else if (auto valueClassValueType = dyn_cast<mlir_ts::ClassValueType>(value.getType()))
        {
            classType = mlir::cast<mlir_ts::ClassType>(valueClassValueType.getClassType());
            classValue = value;
        }
        else
        {
            emitError(location) << "'" << to_print(value.getType()) << "' is not a class, so it can't be a value of '"
                                << to_print(classValueType) << "'";
            return mlir::failure();
        }

        if (classType != baseClassType)
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            auto baseClassInfo = getClassInfoByFullName(baseClassType.getName().getValue());
            std::function<bool(ClassInfo::TypePtr)> extends = [&](ClassInfo::TypePtr derived) {
                return llvm::any_of(derived->baseClasses,
                                    [&](ClassInfo::TypePtr base) { return base == baseClassInfo || extends(base); });
            };
            if (!classInfo || !baseClassInfo || !extends(classInfo))
            {
                emitError(location) << "class '" << to_print(classType) << "' does not extend '" << to_print(baseClassType)
                                    << "', so it can't be a value of '" << to_print(classValueType) << "'";
                return mlir::failure();
            }

            // `new c(args)` passes what the base's constructor takes: the class's must take the same
            auto parametersOf = [&](ClassInfo::TypePtr info) {
                auto *constructorInfo = info->getHasConstructor() ? info->findMethod(CONSTRUCTOR_NAME) : nullptr;
                return constructorInfo ? constructorInfo->funcType.getInputs().drop_front() : llvm::ArrayRef<mlir::Type>();
            };
            auto parameters = parametersOf(classInfo);
            auto baseParameters = parametersOf(baseClassInfo);
            if (parameters.size() > baseParameters.size() ||
                !std::equal(parameters.begin(), parameters.end(), baseParameters.begin()))
            {
                emitError(location) << "the constructor of class '" << to_print(classType)
                                    << "' does not take the arguments the constructor of '" << to_print(baseClassType)
                                    << "' does, so it can't be a value of '" << to_print(classValueType) << "'";
                return mlir::failure();
            }
        }

        if (!classValue)
        {
            auto result = mlirGenClassValue(location, classType, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            classValue = V(result);
        }

        if (classValue.getType() == classValueType)
        {
            return classValue;
        }

        // laid out the same: the base's value is the subclass's
        return V(builder.create<mlir_ts::CastOp>(location, classValueType, classValue));
    }

    mlir::Value MLIRGenImpl::classValuePart(mlir::Location location, mlir::Value classValue, int index)
    {
        auto storageType = MLIRTypeHelper::getClassValueStorageType(builder.getContext());
        auto storage = builder.create<mlir_ts::CastOp>(location, storageType, classValue);
        return builder.create<mlir_ts::ExtractPropertyOp>(location, getOpaqueType(), storage,
                                                          builder.getDenseI64ArrayAttr({index}));
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstanceFromClassValue(mlir::Location location, mlir::Value classValue,
                                                                     NodeArray<Expression> arguments,
                                                                     const GenContext &genContext)
    {
        auto classType = mlir::cast<mlir_ts::ClassType>(mlir::cast<mlir_ts::ClassValueType>(classValue.getType()).getClassType());
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        if (!classInfo)
        {
            emitError(location) << "Can't find class " << to_print(classType);
            return mlir::failure();
        }

        if (classInfo->isAbstract)
        {
            emitError(location) << "Cannot create an instance of an abstract class through a value of '"
                                << to_print(classValue.getType()) << "'";
            return mlir::failure();
        }

        // the arguments, for the constructor the class itself has: a class whose value this can be
        // extends it, and takes what its constructor takes (castToClassValue)
        auto *constructorInfo = classInfo->getHasConstructor() ? classInfo->findMethod(CONSTRUCTOR_NAME) : nullptr;
        if (!constructorInfo && arguments.size() > 0)
        {
            emitError(location) << "Expected 0 arguments, but got " << arguments.size();
            return mlir::failure();
        }

        SmallVector<mlir::Value, 4> operands;
        if (constructorInfo)
        {
            if (mlir::failed(mlirGenOperands(arguments, operands, constructorInfo->funcType, genContext, 1/*this params shift*/)))
            {
                emitError(location) << "Call constructor: can't resolve values of all parameters";
                return mlir::failure();
            }
        }

        // allocate: the class's own `.new`, which sets its vtable
        auto newFuncType = getFunctionType({}, {classType}, false);
        auto newFunc = builder.create<mlir_ts::CastOp>(location, newFuncType, classValuePart(location, classValue, 1));
        auto callNewOp = builder.create<mlir_ts::CallIndirectOp>(location, newFunc, mlir::ValueRange{});
        callNewOp->setAttr(OWNED_RESULT_ATTR_NAME, builder.getUnitAttr());
        callNewOp->setAttr(OWN_FRESH_RESULT_ATTR_NAME, builder.getUnitAttr());
        auto instance = callNewOp.getResult(0);

        // construct: its constructor, when it has one
        auto constructorPart = classValuePart(location, classValue, 2);
        CAST_A(hasConstructor, location, getBooleanType(), constructorPart, genContext);
        // the instance first, as a method's `this` (which the call borrows), then the arguments as a
        // call of the constructor takes them: cast to its parameters, optional ones left out undefined,
        // a rest parameter packed into its array
        auto constructorFuncType = constructorInfo ? constructorInfo->funcType : getFunctionType({classType}, {}, false);
        auto failed = false;
        builder.create<mlir_ts::IfOp>(
            location, hasConstructor, [&](mlir::OpBuilder &opBuilder, mlir::Location loc) {
                auto constructorFunc = builder.create<mlir_ts::CastOp>(location, constructorFuncType, constructorPart);
                auto result = mlirGenCallFunction(location, constructorFuncType, constructorFunc, instance, operands, genContext);
                failed = mlir::failed(result);
                builder.create<mlir_ts::ResultOp>(loc);
            });
        if (failed)
        {
            return mlir::failure();
        }

        return instance;
    }

} // namespace mlirgen
} // namespace typescript
