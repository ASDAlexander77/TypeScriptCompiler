// Variable declaration/binding and identifier resolution methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    // Whether a function in `initializer` (a method, accessor, arrow or function expression) reads the name
    // `name` (`{ m() { return c.v; } }` of `const c: I = ...`). A property's own name (`o.c`, `{ c: 1 }`) is not
    // a read of it, and neither is a function that declares the name itself (a parameter, a local, its own name:
    // `xs.map(v => v + 1)` of `const v`), which reads its own.
    static bool functionInInitializerReadsName(Node initializer, const std::string &name)
    {
        auto found = false;
        auto functionDepth = 0;

        auto isName = [&](Node node) {
            return node == SyntaxKind::Identifier && MLIRHelper::getName(node.as<Identifier>()) == name;
        };

        // whether the function declares the name: a parameter, its own name, or a declaration anywhere in it
        // (a nested one too: telling the scopes apart is not worth it, the variable is then declared as before)
        auto declaresName = [&](Node functionNode) {
            auto declares = false;
            ts::FuncT<> visitDecl;
            ts::ArrayFuncT<> visitDeclArray;
            visitDecl = [&](Node node) -> Node {
                if (declares || !node)
                {
                    return undefined;
                }

                switch ((SyntaxKind)node)
                {
                case SyntaxKind::Parameter:
                    declares = declares || isName(node.as<ParameterDeclaration>()->name);
                    break;
                case SyntaxKind::VariableDeclaration:
                    declares = declares || isName(node.as<VariableDeclaration>()->name);
                    break;
                case SyntaxKind::BindingElement:
                    declares = declares || isName(node.as<BindingElement>()->name);
                    break;
                case SyntaxKind::FunctionDeclaration:
                case SyntaxKind::FunctionExpression:
                case SyntaxKind::ClassDeclaration:
                case SyntaxKind::ClassExpression:
                    declares = declares || isName(node.as<NamedDeclaration>()->name);
                    break;
                default:
                    break;
                }

                ts::forEachChild(node, visitDecl, visitDeclArray);
                return undefined;
            };
            visitDeclArray = [&](NodeArray<Node> array) -> Node {
                for (auto node : array)
                {
                    visitDecl(node);
                }

                return undefined;
            };

            visitDecl(functionNode);
            return declares;
        };

        ts::FuncT<> visitNode;
        ts::ArrayFuncT<> visitArray;

        visitNode = [&](Node node) -> Node {
            if (found || !node)
            {
                return undefined;
            }

            switch ((SyntaxKind)node)
            {
            case SyntaxKind::Identifier:
                if (functionDepth > 0 && isName(node))
                {
                    found = true;
                }

                return undefined;
            case SyntaxKind::PropertyAccessExpression:
                return visitNode(node.as<PropertyAccessExpression>()->expression);
            case SyntaxKind::PropertyAssignment:
                return visitNode(node.as<PropertyAssignment>()->initializer);
            case SyntaxKind::FunctionExpression:
            case SyntaxKind::ArrowFunction:
            case SyntaxKind::MethodDeclaration:
            case SyntaxKind::GetAccessor:
            case SyntaxKind::SetAccessor:
                if (declaresName(node))
                {
                    return undefined;
                }

                // what the function reads: its parameters' defaults and its body, not its own name
                functionDepth++;
                for (auto param : node.as<FunctionLikeDeclarationBase>()->parameters)
                {
                    visitNode(param->initializer);
                }

                visitNode(node.as<FunctionLikeDeclarationBase>()->body);
                functionDepth--;
                return undefined;
            default:
                break;
            }

            ts::forEachChild(node, visitNode, visitArray);
            return undefined;
        };

        visitArray = [&](NodeArray<Node> array) -> Node {
            for (auto node : array)
            {
                visitNode(node);
            }

            return undefined;
        };

        visitNode(initializer);
        return found;
    }

    // `[a, ...rest] = src`: rest is a new array holding src[index..] (#477) - built as
    //   let .rest_n = .rest_src.length - index; if (.rest_n < 0) .rest_n = 0;
    //   .rest = new array(.rest_n); for (let .rest_i = 0; .rest_i < .rest_n; ++.rest_i) .rest[.rest_i] = .rest_src[.rest_i + index];
    // the elements are written into the new array in place, so no second variable holds the same storage
    ValueOrLogicalResult MLIRGenImpl::mlirGenArrayRestCopy(mlir::Location location, mlir_ts::ArrayType arrayType, mlir::Value source,
                                                  int64_t index, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto srcVarDecl = std::make_shared<VariableDeclarationDOM>(".rest_src", arrayType, location);
        DECLARE(srcVarDecl, source);

        NodeFactory nf(NodeFactoryFlags::None);
        auto _src = nf.createIdentifier(S(".rest_src"));
        auto _dst = nf.createIdentifier(S(".rest"));
        auto _n = nf.createIdentifier(S(".rest_n"));
        auto _i = nf.createIdentifier(S(".rest_i"));
        auto _index = [&]() { return nf.createNumericLiteral(stows(std::to_string(index))); };
        auto _zero = [&]() { return nf.createNumericLiteral(S("0")); };

        // let .rest_n = .rest_src.length - index; if (.rest_n < 0) .rest_n = 0;
        NodeArray<VariableDeclaration> countDeclarations;
        countDeclarations.push_back(nf.createVariableDeclaration(
            _n, undefined, undefined,
            nf.createBinaryExpression(nf.createPropertyAccessExpression(_src, nf.createIdentifier(S(LENGTH_FIELD_NAME))),
                                      nf.createToken(SyntaxKind::MinusToken), _index())));
        if (mlir::failed(mlirGen(nf.createVariableStatement(undefined, nf.createVariableDeclarationList(countDeclarations, NodeFlags::Let)), genContext)))
        {
            return mlir::failure();
        }

        auto clamp = nf.createIfStatement(
            nf.createBinaryExpression(_n, nf.createToken(SyntaxKind::LessThanToken), _zero()),
            nf.createExpressionStatement(nf.createBinaryExpression(_n, nf.createToken(SyntaxKind::EqualsToken), _zero())), undefined);
        if (mlir::failed(mlirGen(clamp, genContext)))
        {
            return mlir::failure();
        }

        auto count = resolveIdentifier(location, ".rest_n", genContext);
        auto countAsIndex = cast(location, builder.getIndexType(), count, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(countAsIndex)

        auto newArray = builder.create<mlir_ts::NewArrayOp>(location, arrayType, countAsIndex);
        auto dstVarDecl = std::make_shared<VariableDeclarationDOM>(".rest", arrayType, location);
        DECLARE(dstVarDecl, newArray);

        // for (let .rest_i = 0; .rest_i < .rest_n; ++.rest_i) .rest[.rest_i] = .rest_src[.rest_i + index];
        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, _zero()));
        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken), _n);
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);
        auto assign = nf.createExpressionStatement(nf.createBinaryExpression(
            nf.createElementAccessExpression(_dst, _i), nf.createToken(SyntaxKind::EqualsToken),
            nf.createElementAccessExpression(_src, nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::PlusToken), _index()))));

        if (mlir::failed(mlirGen(nf.createForStatement(initVars, cond, incr, assign), genContext)))
        {
            return mlir::failure();
        }

        return V(newArray);
    }

    ValueOrLogicalResult MLIRGenImpl::registerVariableInThisContext(mlir::Location location, StringRef name, mlir::Type type,
                                                       const GenContext &genContext)
    {
        if (genContext.passResult)
        {

            // create new type with added field
            genContext.passResult->extraFieldsInThisContext.push_back(
                {MLIRHelper::TupleFieldName(name, builder.getContext()), type, false, mlir_ts::AccessLevel::Public});
            return mlir::Value();
        }

        // resolve object property

        NodeFactory nf(NodeFactoryFlags::None);
        // load this.<var name>
        auto _this = nf.createToken(SyntaxKind::ThisKeyword);
        auto _name = nf.createIdentifier(stows(std::string(name)));
        auto _this_name = nf.createPropertyAccessExpression(_this, _name);

        auto result = mlirGen(_this_name, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto thisVarValue = V(result);

        assert(thisVarValue);

        MLIRCodeLogic mcl(builder, compileOptions);
        auto thisVarValueRef = mcl.GetReferenceFromValue(location, thisVarValue);

        assert(thisVarValueRef);

        return V(thisVarValueRef);
    }

    mlir::LogicalResult MLIRGenImpl::registerVariableDeclaration(mlir::Location location, VariableDeclarationDOM::TypePtr variableDeclaration, struct VariableDeclarationInfo &variableDeclarationInfo, bool showWarnings, const GenContext &genContext)
    {
        if (variableDeclarationInfo.deleted)
        {
            return mlir::success();
        }

        if (!variableDeclarationInfo.isGlobal)
        {
            if (mlir::failed(declare(
                location, 
                variableDeclaration, 
                variableDeclarationInfo.storage 
                    ? variableDeclarationInfo.storage 
                    : variableDeclarationInfo.initial, 
                genContext, 
                showWarnings)))
            {
                return mlir::failure();
            }

            if (this->compileOptions.generateDebugInfo 
                && variableDeclarationInfo.initial 
                && !variableDeclarationInfo.storage
                && !isa<mlir_ts::NeverType>(variableDeclarationInfo.initial.getType())
                && !mth.isGenericType(variableDeclarationInfo.initial.getType())
                && !mth.isAnyFunctionType(variableDeclarationInfo.initial.getType())
                // a name of a class, an interface, a namespace or a type (`const k = Box<number>`) has no run-time
                // value to show: its reference has nothing to lower to (TSLANG 171, --di)
                && !variableDeclarationInfo.initial.getDefiningOp<mlir_ts::ClassRefOp>()
                && !variableDeclarationInfo.initial.getDefiningOp<mlir_ts::InterfaceRefOp>()
                && !variableDeclarationInfo.initial.getDefiningOp<mlir_ts::NamespaceRefOp>()
                && !variableDeclarationInfo.initial.getDefiningOp<mlir_ts::TypeRefOp>())
            {
                // to show const values
                MLIRDebugInfoHelper mti(builder, debugScope);
                auto namedLoc = mti.combineWithCurrentScopeAndName(location, variableDeclarationInfo.variableName);
                builder.create<mlir_ts::DebugVariableOp>(namedLoc, variableDeclarationInfo.initial);
            }
        }
        else if (variableDeclarationInfo.isFullName)
        {
            fullNameGlobalsMap.insert(variableDeclarationInfo.fullName, variableDeclaration);
            if (auto *unit = recordingImportUnit())
            {
                unit->globals.push_back({importKeySaver.save(variableDeclarationInfo.fullName), variableDeclaration});
            }
        }
        else
        {
            getGlobalsMap().insert({variableDeclarationInfo.variableName, variableDeclaration});
        }

        return mlir::success();
    }

    // A local that holds a heap reference becomes an owner of it: it takes a reference here and
    // gives it back at every exit from its scope (mlirGenReleaseOwned). Stated unconditionally
    // - `ts.RetainSlot`/`ts.ReleaseSlot` erase whole under a collector, so a collected build
    // sees no trace of this and cannot be broken by where the pair lands.
    //
    // The pair is balanced by construction, which is the property that makes this safe to add
    // before anything consumes a count: the reference an allocation is born with is never given
    // up here, so no release can outnumber its retains. It leaks rather than over-releases -
    // and under `-mm=rc` the collector is still what reclaims, so the leak is inert.
    //
    // Deliberately excluded, each because the frame borrows the reference rather than owning
    // it, and releasing one would drop a count nobody took:
    //  - globals, which outlive every scope;
    //  - parameters, owned by the caller - only variable declarations reach this;
    //  - captured variables held in the `this` context, whose slot belongs to the context;
    //  - const bindings with no storage, which have no slot to release from;
    //  - declarations with no initializer, whose slot holds nothing yet. A catch variable is
    //    the one that matters: it is declared here but written by the landing pad, so
    //    retaining at the declaration would read an uninitialized slot as a live reference.
    //    Consequently a `let s: string;` assigned later never becomes an owner - the
    //    assignment path below only fires on a slot this marked, so that stays balanced;
    //
    // A local declared in a catch or finally clause is owned like any other. It was not, while a
    // call inside an exception funclet was fragile (`catch (e: int) { new Res(); throw 2; }`
    // crashed, since fixed); a borrowing local there read freed memory under rc once its source
    // was given a new value (a fuzzer find, seed 920443).
    //
    // The test itself is localTakesOwnership, shared with the hoisting decision in
    // createLocalVariable so the two cannot disagree about which declarations these are.
    //
    // An excluded declaration is not finished with, though: it may still be captured, and a
    // captured variable's *cell* is the scope's to give back whoever owns the value in it. That
    // is trackPossibleCell below, and it is a different question with a different answer.
    void MLIRGenImpl::takeOwnershipOfLocal(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo,
                                           const GenContext &genContext)
    {
        if (!variableDeclarationInfo.storage || !localTakesOwnership(location, variableDeclarationInfo, genContext))
        {
            trackPossibleCell(variableDeclarationInfo, genContext);
            return;
        }

        auto varOp = variableDeclarationInfo.storage.getDefiningOp<mlir_ts::VariableOp>();
        if (!varOp)
        {
            return;
        }

        varOp->setAttr(OWNED_LOCAL_ATTR_NAME, builder.getUnitAttr());

        // When the initializer already carries a reference for its receiver - `let x = new C()`,
        // whose instance arrives owned (OWNED_RESULT_ATTR_NAME) - taking another would put
        // the slot one owner above the truth and nothing would ever be freed. Consume that
        // reference instead: no retain here, and the scope-exit release is what gives it back.
        // The pair stays balanced, so this cannot over-release; it is only the retain that moves.
        //
        // The attribute is what tells the verifier the release still has a partner, since there
        // is no `ts.RetainSlot` left to pair it with.
        if (producesOwnedReference(variableDeclarationInfo.initial))
        {
            varOp->setAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME, builder.getUnitAttr());
            consumeOwnedReference(variableDeclarationInfo.initial);
        }
        else
        {
            builder.create<mlir_ts::RetainSlotOp>(location, variableDeclarationInfo.storage);
        }

        genContext.ownedVars->push_back(variableDeclarationInfo.storage);
    }

    // A local that owns nothing still has to be listed, because it may yet become a cell.
    //
    // Whether a variable is captured is not known here: the closure that captures it is written
    // after the declaration, and marks the storage when it is generated. Scope exit is generated
    // after both, so that is where the question can be answered - all this does is make sure the
    // slot is there to be asked about. Nothing is emitted for one that never becomes a cell.
    //
    // Two exclusions, and both are about where the release would land rather than about the
    // variable. A declaration directly inside a try body has its release repeated in the
    // cleanup region, which cannot see storage declared in the body - localTakesOwnership pairs
    // its own answer with hoisting for exactly that reason, and hoisting every local of every
    // try body is not a trade this makes. A catch or finally clause is excluded on the same
    // terms takeOwnershipOfLocal excludes it. A captured variable declared in either keeps
    // leaking its cell.
    void MLIRGenImpl::trackPossibleCell(struct VariableDeclarationInfo &variableDeclarationInfo,
                                        const GenContext &genContext)
    {
        if (genContext.ownedVars == nullptr || genContext.allocateScopeOwnedVarsOutsideOfOperation ||
            variableDeclarationInfo.isGlobal || variableDeclarationInfo.deleted ||
            variableDeclarationInfo.allocateInContextThis || blockIsInsideCatchOrFinally())
        {
            return;
        }

        if (!variableDeclarationInfo.storage || !variableDeclarationInfo.storage.getDefiningOp<mlir_ts::VariableOp>())
        {
            return;
        }

        genContext.ownedVars->push_back(variableDeclarationInfo.storage);
    }

    mlir::Type MLIRGenImpl::registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                                TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings, bool forceLocalVar)
    {
        struct VariableDeclarationInfo variableDeclarationInfo(
            compileOptions, func, [this](StringRef name) { return getGlobalsFullNamespaceName(name); });

        variableDeclarationInfo.detectFlags(isFullName, varClass, forceLocalVar, genContext);
        variableDeclarationInfo.setName(name);

        if (declarationMode)
            variableDeclarationInfo.setExternal(true);

        if (!variableDeclarationInfo.isGlobal)
        {
            if (variableDeclarationInfo.isConst) {
                if (mlir::failed(variableDeclarationInfo.processConstRef(location, builder, genContext)))
                    return mlir::Type();

                // a const binding that turns out to need identity storage (see
                // processConstRef / hasBoundMethodField) falls through to the same
                // real-storage path as `let` instead of staying a bare SSA value.
                if (variableDeclarationInfo.needsIdentityStorage
                    && mlir::failed(createLocalVariable(location, variableDeclarationInfo, genContext)))
                    return mlir::Type();

                // Folded: from here the name *is* this value, so a receiver meeting it later
                // cannot know whether it is the last mention - see OWNED_RESULT_NAMED_ATTR_NAME.
                // A ConstRef's initial is a reference to storage and owns nothing of its own.
                if (!variableDeclarationInfo.needsIdentityStorage && compileOptions.tracksOwnership()
                    && variableDeclarationInfo.initial
                    && mth.ownsHeapMemory(location, variableDeclarationInfo.initial.getType()))
                {
                    if (auto *definingOp = variableDeclarationInfo.initial.getDefiningOp())
                    {
                        definingOp->setAttr(OWNED_RESULT_NAMED_ATTR_NAME, builder.getUnitAttr());
                    }
                }
            } else if (mlir::failed(createLocalVariable(location, variableDeclarationInfo, genContext)))
                return mlir::Type();
        }
        else
        {
            variableDeclarationInfo.isSpecialization = genContext.specialization;
            if (mlir::failed(createGlobalVariable(location, variableDeclarationInfo, genContext))) {
                return mlir::Type();
            }

            if (mlir::succeeded(isGlobalConstLambda(location, variableDeclarationInfo, genContext)))
            {
                forgetLastGlobalOf(variableDeclarationInfo.globalOp->getParentOp());
                forgetModuleSymbol(variableDeclarationInfo.globalOp);
                variableDeclarationInfo.globalOp->erase();
                variableDeclarationInfo.deleted = true;
            }
        }

        if (!variableDeclarationInfo.type)
        {
            emitError(location) << "type of variable '" << variableDeclarationInfo.variableName << "' is not valid";
            return variableDeclarationInfo.type;
        }

        // A method of a number, string or other primitive (`n.toFixed`, `let { toFixed } = 1`) is a call on its
        // value; only a local `const` names it (the name is the method with its value). Stored, the call through
        // the variable was dropped (a local) or failed in the lowering ("failed to legalize operation
        // 'ts.CreateExtensionFunction'", a module variable).
        if (isa<mlir_ts::ExtensionFunctionType>(variableDeclarationInfo.type)
            && (variableDeclarationInfo.isGlobal || !variableDeclarationInfo.isConst
                || variableDeclarationInfo.needsIdentityStorage))
        {
            emitError(location) << "a method of a primitive value can't be stored in '"
                                << variableDeclarationInfo.variableName
                                << "'; call it, keep it in a local 'const', or wrap it in an arrow function";
            return mlir::Type();
        }

        // A namespace is no run-time value (TSLANG 107): `var v = M` failed in the lowering ("failed to legalize
        // operation 'ts.NamespaceRef'"). A local `const` is a name for it, as an import alias is.
        if (isa<mlir_ts::NamespaceType>(variableDeclarationInfo.type)
            && variableDeclarationInfo.typeProvided != TypeProvided::Yes
            && (variableDeclarationInfo.isGlobal || !variableDeclarationInfo.isConst
                || variableDeclarationInfo.needsIdentityStorage))
        {
            emitError(location) << "a namespace is not a value and can't be stored in '"
                                << variableDeclarationInfo.variableName << "'; use 'import "
                                << variableDeclarationInfo.variableName << " = <namespace>' for another name of it";
            return mlir::Type();
        }

        // remembered on the storage for warnIfIntegerLiteralVariableTruncates
        if (variableDeclarationInfo.inferredFromIntegerLiteral && !variableDeclarationInfo.deleted)
        {
            auto varOp = variableDeclarationInfo.storage ? variableDeclarationInfo.storage.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            mlir::Operation *storageOp = varOp ? varOp.getOperation() : variableDeclarationInfo.globalOp.getOperation();
            if (storageOp)
            {
                storageOp->setAttr(INTEGER_LITERAL_INFERRED_ATTR_NAME, builder.getUnitAttr());
            }
        }

        //LLVM_DEBUG(variableDeclarationInfo.printDebugInfo(););

        takeOwnershipOfLocal(location, variableDeclarationInfo, genContext);

        auto varDecl = variableDeclarationInfo.createVariableDeclaration(location, genContext);
        if (genContext.usingVars != nullptr && varDecl->getUsing())
        {
            varDecl->setDisposeGuard(variableDeclarationInfo.disposeGuard);
            genContext.usingVars->push_back(varDecl);
        }

        registerVariableDeclaration(location, varDecl, variableDeclarationInfo, showWarnings, genContext);
        return varDecl->getType();
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationArrayBindingPatternSubPath(
        mlir::Location location, int index, mlir::Type type, mlir::Value init, 
        bool isDotDotDot, bool isIterator, bool isArrayLike, mlir::Value arrayLikeLengthValue, mlir::Type arrayLikeElementType,
        Expression defaultValue, bool bindsName, const GenContext &genContext)
    {
        // An element's default (`const [a = mk(1), b = mk(2)] = [mk(i)]`) is its value where the element is missing - past
        // the end of the source, or undefined there - as in TypeScript. It was not applied: an element past the end of a
        // literal was undefined (the compiler crashed reading it) or the default of its type (`[a = 1, b = 2] = [5]` gave
        // b 0). `castTo`: the element type the value must have, where the other way gives one of that type.
        auto makeDefault = [&](mlir::Type castTo) -> mlir::Value {
            auto result = mlirGen(defaultValue, genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Value();
            }

            auto value = V(result);
            if (castTo && value.getType() != castTo)
            {
                auto castResult = cast(location, castTo, value, genContext);
                if (castResult.failed_or_no_value())
                {
                    return mlir::Value();
                }

                value = V(castResult);
            }

            return value;
        };

        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, init, builder.getI32IntegerAttr(index));
        mlir::Value subInit =
            mlir::TypeSwitch<mlir::Type, mlir::Value>(type)
                .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { 
                    if (isDotDotDot)
                    {
                        SmallVector<mlir::Value> arrayValues;
                        SmallVector<mlir_ts::FieldInfo> fieldInfos;

                        SmallVector<mlir_ts::FieldInfo> srcFieldInfos;
                        if (mlir::failed(mth.getFields(constTupleType, srcFieldInfos)))
                        {
                            return mlir::Value();
                        }

                        for (auto indexSpread = index; indexSpread < srcFieldInfos.size(); indexSpread++)                        
                        {
                            MLIRPropertyAccessCodeLogic cl(
                                compileOptions, builder, location, init, builder.getI32IntegerAttr(indexSpread));
                            auto value = cl.Tuple(constTupleType, true);

                            // the field as it was read: `baz?: any` reads as `any | undefined`, which a field typed
                            // `any` took uncast, and lowering could not box it (no `typeof` name)
                            auto fieldInfo = srcFieldInfos[indexSpread];
                            fieldInfo.type = value.getType();
                            fieldInfos.push_back(fieldInfo);
                            arrayValues.push_back(value);
                        }
                
                        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues));
                    }

                    // past the end of the tuple (`var [a, b, c] = [1, "x"]`, a literal taken as a tuple): undefined,
                    // as the element of an array is; read, it was no value and the statement failed with nothing said
                    if (index >= (int)constTupleType.size())
                    {
                        return defaultValue ? makeDefault(mlir::Type()) : V(builder.create<mlir_ts::UndefOp>(location, getUndefinedType()));
                    }

                    return cl.Tuple(constTupleType, true); 
                })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) { 
                    if (isDotDotDot)
                    {
                        SmallVector<mlir::Value> arrayValues;
                        SmallVector<mlir_ts::FieldInfo> fieldInfos;

                        SmallVector<mlir_ts::FieldInfo> srcFieldInfos;
                        if (mlir::failed(mth.getFields(tupleType, srcFieldInfos)))
                        {
                            return mlir::Value();
                        }

                        for (auto indexSpread = index; indexSpread < srcFieldInfos.size(); indexSpread++)                        
                        {
                            MLIRPropertyAccessCodeLogic cl(
                                compileOptions, builder, location, init, builder.getI32IntegerAttr(indexSpread));
                            auto value = cl.Tuple(tupleType, true);

                            // the field as it was read: `baz?: any` reads as `any | undefined`, which a field typed
                            // `any` took uncast, and lowering could not box it (no `typeof` name)
                            auto fieldInfo = srcFieldInfos[indexSpread];
                            fieldInfo.type = value.getType();
                            fieldInfos.push_back(fieldInfo);
                            arrayValues.push_back(value);
                        }
                
                        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues));
                    }

                    // past the end of the tuple (`var [a, b, c] = [1, "x"]`, a literal taken as a tuple): undefined,
                    // as the element of an array is; read, it was no value and the statement failed with nothing said
                    if (index >= (int)tupleType.size())
                    {
                        return defaultValue ? makeDefault(mlir::Type()) : V(builder.create<mlir_ts::UndefOp>(location, getUndefinedType()));
                    }

                    return cl.Tuple(tupleType, true); 
                })
                .template Case<mlir_ts::ConstArrayType>([&](auto constArrayType) {
                    if (isDotDotDot)
                    {   
                        auto arrayType = mth.removeConstType(constArrayType);

                        auto arrayValue = cast(location, arrayType, init, genContext);
                        if (!arrayValue)
                        {
                            return mlir::Value();
                        }

                        return V(mlirGenArrayRestCopy(location, mlir::cast<mlir_ts::ArrayType>(arrayType), arrayValue, index, genContext));
                    }

                    // past the end: the default of the element type, null for a reference (see below)
                    if (static_cast<unsigned>(index) >= constArrayType.getSize())
                    {
                        return defaultValue ? makeDefault(mlir::Type())
                                            : V(builder.create<mlir_ts::DefaultOp>(location, constArrayType.getElementType()));
                    }

                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                        builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                        location, mlir_ts::RefType::get(constArrayType.getElementType()), init, constIndex);
                    return V(builder.create<mlir_ts::LoadOp>(location, constArrayType.getElementType(), elemRef));
                })
                .template Case<mlir_ts::ArrayType>([&](auto arrayType) {

                    if (isDotDotDot)
                    {   
                        return V(mlirGenArrayRestCopy(location, arrayType, init, index, genContext));
                    }

                    // An element past the end of the array (`const [a, b, c] = pair`) is the default
                    // of its type - null for a reference, where TS has undefined - not whatever follows
                    // the array's data: a read of that was undefined behaviour, and under rc a count
                    // taken of it faulted.
                    auto lengthValue = builder.create<mlir_ts::LengthOfOp>(location, builder.getIndexType(), init);
                    auto indexValue = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                                          mth.getIndexAttrValue(index));
                    auto inBoundsValue = V(builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                        builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanToken), lengthValue, indexValue));

                    // (a literal that has the element - `[[k = mk(1)] = []] = [[mk(i)]]` - always gives it: the default
                    // never applies, as for a constant array)
                    // (unless the element may be undefined: `[u = 3] = [undefined]`)
                    auto elementMayBeUndefined = [&](mlir::Type elementType) {
                        if (isa<mlir_ts::UndefinedType, mlir_ts::OptionalType, mlir_ts::AnyType, mlir_ts::UnknownType>(elementType))
                        {
                            return true;
                        }

                        auto unionType = dyn_cast<mlir_ts::UnionType>(elementType);
                        return unionType && llvm::any_of(unionType.getTypes(), [](mlir::Type type) {
                                   return isa<mlir_ts::UndefinedType, mlir_ts::OptionalType>(type);
                               });
                    };
                    // (also an inner literal read out of an outer one: `[[b = mk(1)] = [mk(2)]] = [[mk(i)]]`)
                    auto literalOf = [](mlir::Value value) {
                        while (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
                        {
                            auto elementRefOp = loadOp.getReference().getDefiningOp<mlir_ts::ElementRefOp>();
                            auto outer = elementRefOp ? elementRefOp->getOperand(0).getDefiningOp<mlir_ts::CreateArrayOp>()
                                                      : mlir_ts::CreateArrayOp();
                            auto indexOp = elementRefOp ? elementRefOp->getOperand(1).getDefiningOp<mlir_ts::ConstantOp>()
                                                        : mlir_ts::ConstantOp();
                            auto indexAttr = indexOp ? dyn_cast<mlir::IntegerAttr>(indexOp.getValue()) : mlir::IntegerAttr();
                            if (!outer || !indexAttr || indexAttr.getInt() < 0 ||
                                indexAttr.getInt() >= (int64_t)outer->getNumOperands())
                            {
                                break;
                            }

                            value = outer->getOperand(indexAttr.getInt());
                        }

                        return value.getDefiningOp<mlir_ts::CreateArrayOp>();
                    };
                    if (auto createArrayOp = literalOf(init);
                        defaultValue && createArrayOp && static_cast<unsigned>(index) < createArrayOp->getNumOperands() &&
                        !elementMayBeUndefined(arrayType.getElementType()))
                    {
                        // the element, read as out of a constant array: no bounds test (an inner pattern reads it on)
                        auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                            builder.getI32IntegerAttr(index));
                        auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                            location, mlir_ts::RefType::get(arrayType.getElementType()), init, constIndex);
                        return V(builder.create<mlir_ts::LoadOp>(location, arrayType.getElementType(), elemRef));
                    }

                    // (with a default of another type, `const [a, b = "q"] = numbers`, the element is of both)
                    mlir::Type valueType = arrayType.getElementType();
                    if (defaultValue)
                    {
                        if (auto defaultType = evaluate(defaultValue, genContext))
                        {
                            defaultType = mth.wideStorageType(defaultType);
                            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr, mlir::Type>> typeParamsWithArgs;
                            // (an array default of an array element - `[[k = mk(1)] = []]` - is of the element's type)
                            auto bothArrays = isa<mlir_ts::ArrayType, mlir_ts::ConstArrayType>(defaultType) &&
                                              isa<mlir_ts::ArrayType>(valueType);
                            if (defaultType != valueType && !bothArrays &&
                                mth.extendsType(location, defaultType, valueType, typeParamsWithArgs) != ExtendsResult::True)
                            {
                                valueType = getUnionType(location, valueType, defaultType);
                            }
                        }
                    }

                    // (without a default, an element past the end is undefined: a name bound to an element of a reference
                    // type is `T | undefined` - readsUndefinedPastEnd; it read as null. Not where a literal has the element,
                    // nor for an inner pattern, which reads the element on)
                    auto createArrayOfInit = literalOf(init);
                    if (!defaultValue && !isDotDotDot && bindsName && readsUndefinedPastEnd(valueType) &&
                        !(createArrayOfInit && static_cast<unsigned>(index) < createArrayOfInit->getNumOperands()))
                    {
                        valueType = mlir_ts::OptionalType::get(valueType);
                    }

                    // -mm=own: a default that makes a new value is held by a slot of the scope, as an object pattern's is
                    // (optionalValueOrDefault); the element read borrows the slot or the source's element. A new value
                    // merged with a borrow of the element had no owner: own refused the pattern.
                    auto defaultSlot = defaultValue ? ownPatternDefaultSlot(location, valueType, genContext) : mlir::Value();

                    auto result = conditionalValue(location, inBoundsValue,
                        [&]() {
                            // TODO: unify it with ElementAccess
                            auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                                builder.getI32IntegerAttr(index));
                            auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                                location, mlir_ts::RefType::get(arrayType.getElementType()), init, constIndex);
                            mlir::Value element = V(builder.create<mlir_ts::LoadOp>(location, arrayType.getElementType(), elemRef));
                            if (element.getType() != valueType)
                            {
                                auto castResult = cast(location, valueType, element, genContext);
                                EXIT_IF_FAILED_OR_NO_VALUE(castResult)
                                element = V(castResult);
                            }

                            return ValueOrLogicalResult(element);
                        },
                        [&](mlir::Type elementType) {
                            if (defaultValue)
                            {
                                auto value = makeDefault(elementType);
                                if (!value)
                                {
                                    return ValueOrLogicalResult(mlir::failure());
                                }

                                // (kept only for a new value: settleOwnPatternDefaults)
                                if (defaultSlot && value.getType() == valueType)
                                {
                                    builder.create<mlir_ts::StoreOp>(location, value, defaultSlot);
                                    value = builder.create<mlir_ts::LoadOp>(location, valueType, defaultSlot);
                                }

                                return ValueOrLogicalResult(value);
                            }

                            if (isa<mlir_ts::OptionalType>(elementType))
                            {
                                return ValueOrLogicalResult(V(builder.create<mlir_ts::OptionalUndefOp>(location, elementType)));
                            }

                            return ValueOrLogicalResult(V(builder.create<mlir_ts::DefaultOp>(location, elementType)));
                        });
                    return result.failed_or_no_value() ? mlir::Value() : V(result);
                })
                .Default([&](auto type) { 

                    if (isDotDotDot)
                    {
                        emitError(location) << "Array Binding Pattern: spread is not implemented for type: " << to_print(type);
                        return mlir::Value();  
                    }

                    if (isIterator)
                    {
                        // seems it is "iterator"
                        auto nextProperty = init;
                        auto result = callIteratorNext(location, nextProperty, nullptr, genContext);
                        return V(result);
                    }

                    // add array like access
                    if (isArrayLike)
                    {
                        auto valueFactory =
                        (isa<mlir_ts::AnyType>(arrayLikeElementType))
                            ? &MLIRGenImpl::anyOrUndefined
                            : &MLIRGenImpl::optionalValueOrUndefined;
            
                        auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                            mth.getIndexAttrValue(index));
        
                        // conditional expr: length > "spreadIndex" ? value[index] : undefined
                        auto inBoundsValue = V(builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                            builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanToken), 
                            arrayLikeLengthValue,
                            indexVal));
        
                        auto spreadValue = (this->*valueFactory)(location, inBoundsValue,
                            [&](auto genContext) {
                                auto result = mlirGenElementAccess(location, init, indexVal, false, genContext);
                                EXIT_IF_FAILED_OR_NO_VALUE(result)
                                return result;
                            }, genContext);
                        //EXIT_IF_FAILED_OR_NO_VALUE(spreadValue)
                        return V(spreadValue);
                    }

                    emitError(location) << "Array Binding Pattern: not implemented for type: " << to_print(type);
                    return mlir::Value(); 
                });

        if (!subInit)
        {
            return mlir::failure();
        }

        // ...and where the element read is undefined, or may be (`[a = 1] = [undefined]`, an optional element)
        if (defaultValue && !isDotDotDot)
        {
            if (isa<mlir_ts::UndefinedType>(subInit.getType()))
            {
                return makeDefault(mlir::Type());
            }

            if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(subInit.getType()))
            {
                auto result = optionalValueOrDefault(location, optionalType.getElementType(), subInit, defaultValue, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                return V(result);
            }
        }

        return subInit; 
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationArrayBindingPattern(mlir::Location location, ArrayBindingPattern arrayBindingPattern,
                                               VariableClass varClass,
                                               TypeValueInitFuncType func,
                                               const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

        if (!init)
        {
            return mlir::failure();
        }

        mlir::Value arrayLikeLengthValue;
        mlir::Type arrayLikeElementType;
        auto isIterator = false;
        auto isSourceArrayLike = false;
        auto isArrayOrTuple = isa<mlir_ts::ArrayType>(typeRef)
            || isa<mlir_ts::TupleType>(typeRef)
            || isa<mlir_ts::ConstArrayType>(typeRef)
            || isa<mlir_ts::ConstTupleType>(typeRef);
        if (!isArrayOrTuple)
        {
            if (auto iteratorType = evaluateProperty(location, init, SYMBOL_ITERATOR, genContext))
            {
                if (auto iteratorResult = mlirGenCallThisMethod(location, init, SYMBOL_ITERATOR, undefined, undefined, genContext))
                {
                    auto iteratorValue = V(iteratorResult);

                    // request iterator
                    auto nextProperty = mlirGenPropertyAccessExpression(
                        location, iteratorValue, ITERATOR_NEXT, false, genContext);
                    if (nextProperty)
                    {
                        init = V(nextProperty);
                        isIterator = true;
                    }                    
                }
            }
            else if (hasIterator(location, init, genContext))
            {
                // request iterator
                auto nextProperty = mlirGenPropertyAccessExpression(
                    location, init, ITERATOR_NEXT, false, genContext);
                if (nextProperty)
                {
                    init = V(nextProperty);
                    isIterator = true;
                }
            }
            else if (isArrayLike(location, init, genContext))
            {
                auto lengthValue = mlirGenPropertyAccessExpression(location, init, LENGTH_FIELD_NAME, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(lengthValue)
                arrayLikeLengthValue = V(lengthValue);
                CAST(arrayLikeLengthValue, location, builder.getIndexType(), arrayLikeLengthValue, genContext);
                auto elementType = evaluateElementAccess(location, init, false, genContext);
                if (elementType)
                {
                    isSourceArrayLike = true;
                    arrayLikeElementType = elementType;
                }
            }            
            else
            {
                emitError(location) << "an array pattern ('[a, b] = ...') of a value of type '" << to_print(init.getType())
                                    << "' is not supported: give it an array, a tuple or an iterable";
                return mlir::failure();
            }
        }        

        for (auto [index, element] : enumerate(arrayBindingPattern->elements))
        {
            if (element == SyntaxKind::OmittedExpression)
            {
                continue;
            }

            if (element != SyntaxKind::BindingElement)
            {
                emitError(location) << "Array Binding Pattern: unsupported element";
                return mlir::failure();
            }

            auto arrayBindingElement = element.as<BindingElement>();

            auto subValueFunc = [&](mlir::Location location, const GenContext &genContext) { 
                auto result = processDeclarationArrayBindingPatternSubPath(
                    location, index, type, init, !!arrayBindingElement->dotDotDotToken, isIterator, isSourceArrayLike, arrayLikeLengthValue, arrayLikeElementType,
                    arrayBindingElement->initializer, arrayBindingElement->name == SyntaxKind::Identifier, genContext);
                if (result.failed_or_no_value()) 
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No); 
                }

                auto value = V(result);
                return std::make_tuple(value.getType(), value, TypeProvided::No); 
            };

            if (mlir::failed(processDeclaration(
                    arrayBindingElement, varClass, subValueFunc, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationObjectBindingPatternSubPath(
        mlir::Location location, BindingElement objectBindingElement, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        auto fieldName = getFieldNameFromBindingElement(objectBindingElement);
        auto isNumericAccess = isa<mlir::IntegerAttr>(fieldName);

        LLVM_DEBUG(llvm::dbgs() << "ObjectBindingPattern:\n\t" << init << "\n\tprop: " << fieldName << "\n");

        mlir::Value subInit;
        mlir::Type subInitType;

        mlir::Value value;
        if (isNumericAccess)
        {
            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, init, fieldName);
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                value = cl.Tuple(tupleType, true);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                value = cl.Tuple(constTupleType, true);
            }
            else if (isa<mlir_ts::ArrayType>(type) || isa<mlir_ts::ConstArrayType>(type))
            {
                // `const { 0: a } = [0, 1, 2]` reads the element, as `[0, 1, 2][0]`
                auto indexValue = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), fieldName);
                auto result = mlirGenElementAccess(location, init, indexValue, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                value = V(result);
            }
            else
            {
                emitError(location) << "a numeric property name can't be read from a value of type '" << to_print(type)
                                    << "'";
                return mlir::failure();
            }
        }
        else
        {
            auto result = mlirGenPropertyAccessExpression(location, init, fieldName, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            value = V(result);
        }

        if (!value)
        {
            return mlir::failure();
        }

        // the default applies to a field that may be missing; one the source always has (`var { x = 500 } = { x: 20 }`,
        // a const tuple) keeps its value, as in TypeScript, where the default is then never used
        mlir_ts::OptionalType optionalFieldType = dyn_cast<mlir_ts::OptionalType>(value.getType());
        if (!optionalFieldType)
        {
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                auto index = tupleType.getIndex(fieldName);
                if (index >= 0)
                {
                    optionalFieldType = dyn_cast<mlir_ts::OptionalType>(tupleType.getFieldInfo(index).type);
                }
            }
        }

        if (objectBindingElement->initializer && optionalFieldType)
        {
            auto subType = optionalFieldType.getElementType();
            auto res = optionalValueOrDefault(location, subType, value, objectBindingElement->initializer, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(res)
            subInit = V(res);
            subInitType = subInit.getType();
        }
        else
        {
            subInit = value;
            subInitType = subInit.getType();
        }

        assert(subInit);

        return subInit; 
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationObjectBindingPatternSubPathSpread(
        mlir::Location location, ObjectBindingPattern objectBindingPattern, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        mlir::Value subInit;
        mlir::Type subInitType;

        // the rest of an array is an object of its other indices and `length`; nothing here makes one
        // (it was cast to an empty tuple, with an "invalid cast" warning)
        if (isa<mlir_ts::ArrayType>(init.getType()) || isa<mlir_ts::ConstArrayType>(init.getType()))
        {
            emitError(location) << "the rest element of an object pattern can't take the rest of an array: "
                                << "use an array pattern, `[a, ...b]`";
            return mlir::failure();
        }

        SmallVector<mlir::Attribute> names;

        // take all used fields
        for (auto objectBindingElement : objectBindingPattern->elements)
        {
            auto isSpreadBinding = !!objectBindingElement->dotDotDotToken;
            if (isSpreadBinding)
            {
                continue;
            }

            auto fieldId = getFieldNameFromBindingElement(objectBindingElement);
            names.push_back(fieldId);
        }                

        // filter all fields
        llvm::SmallVector<mlir_ts::FieldInfo> tupleFields;
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        if (mlir::succeeded(mth.getFields(init.getType(), tupleFields)))
        {
            for (auto fieldInfo : tupleFields)
            {
                if (std::find_if(names.begin(), names.end(), [&] (auto& item) { return item == fieldInfo.id; }) == names.end())
                {
                    // filter;
                    destTupleFields.push_back(fieldInfo);
                }
            }
        }

        // create object
        subInitType = getTupleType(destTupleFields);
        CAST(subInit, location, subInitType, init, genContext);

        assert(subInit);

        return subInit; 
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationObjectBindingPattern(mlir::Location location, ObjectBindingPattern objectBindingPattern,
                                                VariableClass varClass,
                                                TypeValueInitFuncType func,
                                                const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

        // the value failed: its error is out already (each field read from no value crashed)
        if (!init)
        {
            return mlir::failure();
        }

        for (auto objectBindingElement : objectBindingPattern->elements)
        {
            auto subValueFunc = [&] (mlir::Location location, const GenContext &genContext) {

                auto isSpreadBinding = !!objectBindingElement->dotDotDotToken;
                auto result = isSpreadBinding 
                    ? processDeclarationObjectBindingPatternSubPathSpread(location, objectBindingPattern, type, init, genContext)
                    : processDeclarationObjectBindingPatternSubPath(location, objectBindingElement, type, init, genContext);
                if (result.failed_or_no_value()) 
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No); 
                }                    

                auto value = V(result);
                return std::make_tuple(value.getType(), value, TypeProvided::No); 
            };

            // nested obj, objectBindingElement->propertyName -> name
            if (objectBindingElement->name == SyntaxKind::ObjectBindingPattern)
            {
                auto objectBindingPattern = objectBindingElement->name.as<ObjectBindingPattern>();

                return processDeclarationObjectBindingPattern(
                    location, objectBindingPattern, varClass, subValueFunc, genContext);
            }

            if (mlir::failed(processDeclaration(
                    objectBindingElement, varClass, subValueFunc, genContext)))
            { 
                return mlir::failure();
            }
        }

        return mlir::success();;
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationName(DeclarationName name, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings)
    {
        auto location = loc(name);

        if (name == SyntaxKind::ArrayBindingPattern)
        {
            auto arrayBindingPattern = name.as<ArrayBindingPattern>();
            return processDeclarationArrayBindingPattern(location, arrayBindingPattern, varClass, func, genContext);
        }
        else if (name == SyntaxKind::ObjectBindingPattern)
        {
            auto objectBindingPattern = name.as<ObjectBindingPattern>();
            return processDeclarationObjectBindingPattern(location, objectBindingPattern, varClass, func, genContext);
        }
        else
        {
            // name
            auto nameStr = MLIRHelper::getName(name);

            // register
            auto varType = registerVariable(location, nameStr, false, varClass, func, genContext, showWarnings);
            if (!varType)
            {
                return mlir::failure();
            }

            if (varClass.isExport)
            {
                auto isConst = varClass.type == VariableType::Const || varClass.type == VariableType::ConstRef;
                addVariableDeclarationToExport(nameStr, currentNamespace, varType, isConst, varClass.dllName);
            }

            return mlir::success();
        }

        return mlir::failure();       
    }

    mlir::LogicalResult MLIRGenImpl::processDeclaration(NamedDeclaration item, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings)
    {
        if (item == SyntaxKind::OmittedExpression)
        {
            return mlir::success();
        }

        item->name->parent = item;
        return processDeclarationName(item->name, varClass, func, genContext, showWarnings);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(VariableDeclaration item, VariableClass varClass, const GenContext &genContext)
    {
        // a local of the function generated, in a generator's next() too (InternalFlags::StackLocal)
        if ((item->internalFlags & InternalFlags::StackLocal) == InternalFlags::StackLocal && genContext.allocateVarsInContextThis)
        {
            GenContext localGenContext(genContext);
            localGenContext.allocateVarsInContextThis = false;
            return mlirGen(item, varClass, localGenContext);
        }

        auto location = loc(item);

#ifndef ANY_AS_DEFAULT
        auto isExternal = varClass == VariableType::External;
        if (declarationMode)
        {
            isExternal = true;
        }

        if (mth.isNoneType(item->type) && !item->initializer && !isExternal)
        {
            auto name = MLIRHelper::getName(item->name);
            emitError(loc(item)) << "type of variable '" << name
                                 << "' is not provided, variable must have type or initializer";
            return mlir::failure();
        }
#endif

        // `declare const c: any = 0`: an ambient variable is defined elsewhere, and code for its initializer had no
        // function to go in - it went into the module itself, and LLVM's translation crashed on it. tsc allows only
        // a string or numeric literal or a literal enum reference, with no type (TS1039, TS1254).
        if (varClass == VariableType::External && item->initializer)
        {
            auto initializer = item->initializer;
            auto isNumericOrStringLiteral = [](Expression expr) {
                if (expr == SyntaxKind::PrefixUnaryExpression
                    && expr.as<PrefixUnaryExpression>()->_operator == SyntaxKind::MinusToken)
                {
                    expr = expr.as<PrefixUnaryExpression>()->operand;
                }

                // ... or a literal enum member reference (`Bar.a`, `Bar['b']`)
                return expr == SyntaxKind::NumericLiteral || expr == SyntaxKind::StringLiteral
                    || expr == SyntaxKind::NoSubstitutionTemplateLiteral || expr == SyntaxKind::BigIntLiteral
                    || expr == SyntaxKind::PropertyAccessExpression || expr == SyntaxKind::ElementAccessExpression;
            };

            if (item->type || !isNumericOrStringLiteral(initializer))
            {
                emitError(loc(initializer)) << "initializers are not allowed in ambient contexts, except a string or "
                                               "numeric literal of a 'const' with no type";
                return mlir::failure();
            }
        }

        if (!item->type && item->initializer == SyntaxKind::ObjectLiteralExpression)
        {
            item->initializer->internalFlags |= InternalFlags::UntypedVariableInitializer;
        }

        auto initFunc = [&](mlir::Location location, const GenContext &genContext) {
            // A module imported as source only declares what it defines, so a module-level
            // variable gets its type and no initializer code. Not a local: a function body is only
            // walked here to infer types (a dummy run, see mlirGenFunctionBody) and its IR is
            // discarded, so its initializers may produce real values - and must. A `const` binds
            // its name to that value, and a destructuring pattern reads its elements from it:
            // without one, reading the const failed with "can't resolve name" and destructuring
            // with "failed statement" or a crash.
            if (declarationMode && !genContext.funcOp)
            {
                auto [t, b, p] = evaluateTypeAndInit(item, genContext);
                // `export var y;` - no type, no value - is an `any`, as where the module is compiled itself (its
                // importer failed with "failed statement", TSLANG 191)
                if (!t && !item->type && !item->initializer)
                {
                    t = getAnyType();
                }

                return std::make_tuple(t, mlir::Value(), p ? TypeProvided::Yes : TypeProvided::No);
            }

            auto typeAndInit = getTypeAndInit(item, genContext);

            // A generic function value can only stay generic in a `const`, which binds the name to the
            // function itself. A `let`/`var` holds one function, instantiated with `any`. Kept generic,
            // a module variable named a function that was never instantiated (TSLANG 106) and a local
            // one crashed.
            if (auto initValue = std::get<1>(typeAndInit);
                initValue && std::get<2>(typeAndInit) == TypeProvided::No
                && varClass.type != VariableType::Const && varClass.type != VariableType::ConstRef
                && (item->internalFlags & InternalFlags::ForceConst) != InternalFlags::ForceConst
                && (item->internalFlags & InternalFlags::ForceConstRef) != InternalFlags::ForceConstRef
                && isGenericFunctionReference(initValue))
            {
                auto instantiated = instantiateGenericFunctionWithAny(location, initValue, genContext);
                if (instantiated.failed_or_no_value())
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No);
                }

                auto instantiatedValue = V(instantiated);
                return std::make_tuple(instantiatedValue.getType(), instantiatedValue, TypeProvided::No);
            }

            // `let k = C`: the class's value (a local `const` keeps the name of the class itself; a module one is a
            // global, which a class's name can't be stored in: "failed to legalize operation 'ts.ClassRef'",
            // TSLANG 172)
            auto isLocalConst = genContext.funcOp
                && (varClass.type == VariableType::Const || varClass.type == VariableType::ConstRef
                    || (item->internalFlags & InternalFlags::ForceConst) == InternalFlags::ForceConst
                    || (item->internalFlags & InternalFlags::ForceConstRef) == InternalFlags::ForceConstRef);
            if (auto initValue = std::get<1>(typeAndInit);
                initValue && std::get<2>(typeAndInit) == TypeProvided::No && !isLocalConst
                && varClass.type != VariableType::External && !declarationMode
                && initValue.getDefiningOp<mlir_ts::ClassRefOp>())
            {
                auto classValue = classRefAsClassValue(location, initValue, genContext);
                if (classValue.failed_or_no_value())
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No);
                }

                return std::make_tuple(V(classValue).getType(), V(classValue), TypeProvided::No);
            }

            if (varClass.isDynamicImport)
            {
                auto nameStr = varClass.dllName.empty()
                    ? concatFullNamespaceName(MLIRHelper::getName(item->name))
                    : varClass.dllName.str();
                auto fieldType = std::get<0>(typeAndInit);
                if (fieldType)
                {
                    auto dllVarName = V(mlirGenStringValue(location, nameStr, true));
                    auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                        location, getOpaqueType(), dllVarName);

                    // The variable is the library's own: the global holds its address, and every
                    // read and write goes through it (createVariableDeclaration makes the name
                    // indirect, globalVariableAccess loads through it). A copy loaded here was a
                    // snapshot of its value when the program started, which never saw what the
                    // library assigned to it later.
                    auto storageType = fieldType;
                    if (varClass.isBoxed)
                    {
                        // the library's storage is a single boxed pointer (ObjectType lowers to
                        // `ptr`) to the real data, not the data inline. See DeclarationPrinter.cpp's
                        // @boxed emission.
                        storageType = getObjectType(fieldType);
                    }
                    else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(fieldType); funcType && !funcType.isCVarArg())
                    {
                        // an exported global holds a function as a hybrid one (adjustGlobalVariableType)
                        storageType = mlir_ts::HybridFunctionType::get(builder.getContext(), funcType);
                    }

                    auto refToStorage = cast(location, mlir_ts::RefType::get(storageType), referenceToStaticFieldOpaque, genContext);
                    if (refToStorage.failed_or_no_value())
                    {
                        return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No);
                    }

                    return std::make_tuple(V(refToStorage).getType(), V(refToStorage), TypeProvided::Yes);
                }
            }

            return typeAndInit;
        };

        auto valClassItem = varClass;
        if ((item->internalFlags & InternalFlags::ForceConst) == InternalFlags::ForceConst)
        {
            valClassItem = VariableType::Const;
        }

        if ((item->internalFlags & InternalFlags::ForceConstRef) == InternalFlags::ForceConstRef)
        {
            valClassItem = VariableType::ConstRef;
        }

        if (!genContext.funcOp && (item->name == SyntaxKind::ObjectBindingPattern || item->name == SyntaxKind::ArrayBindingPattern))
        {
            auto name = MLIRHelper::getAnonymousName(location, ".gc", "");
            auto fullInitGlobalFuncName = getFullNamespaceName(name);

            {
                mlir::OpBuilder::InsertionGuard insertGuard(builder);

                // create global construct
                valClassItem = VariableType::Var;

                auto funcType = getFunctionType({}, {}, false);

                if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, funcType,
                    [&](mlir::Location location, const GenContext &genContext) {
                        return processDeclaration(item, valClassItem, initFunc, genContext, true);
                    }, genContext)))
                {
                    return mlir::failure();
                }

                addGlobalConstructor(location, fullInitGlobalFuncName, /*afterGlobals=*/true);
            }
        }
        else if (genContext.funcOp && !declarationMode && item->type && item->initializer
                 && item->name == SyntaxKind::Identifier
                 && (varClass.type == VariableType::Const || varClass.type == VariableType::Let
                     || varClass.type == VariableType::Var)
                 && functionInInitializerReadsName(item->initializer, MLIRHelper::getName(item->name)))
        {
            // a local with a type written whose initializer has a function reading it (`const c: I = { m() {
            // return c.v; } }`, TSLANG 148): it is declared first, as a `let` of that type, and then given the
            // initializer, so the function captures the variable, not a value it does not have yet
            auto writtenType = getType(item->type, genContext);
            if (!writtenType)
            {
                return mlir::failure();
            }

            auto declareOnly = [&](mlir::Location, const GenContext &) {
                return std::make_tuple(writtenType, mlir::Value(), TypeProvided::Yes);
            };

            VariableClass declaredFirstClass = varClass;
            declaredFirstClass.type = VariableType::Let;
            if (mlir::failed(processDeclaration(item, declaredFirstClass, declareOnly, genContext, true)))
            {
                return mlir::failure();
            }

            NodeFactory nf(NodeFactoryFlags::None);
            auto assignment = nf.createBinaryExpression(item->name, nf.createToken(SyntaxKind::EqualsToken), item->initializer);
            auto result = mlirGen(assignment, genContext);
            EXIT_IF_FAILED(result)

            // a `const` is assigned only here
            if (varClass.type == VariableType::Const)
            {
                if (auto declared = symbolTable.lookup(MLIRHelper::getName(item->name)); declared.second)
                {
                    declared.second->setConstAssignedOnce();
                }
            }
        }
        else
        {
            // a module variable with a type written is that type in its own initializer: a function there may
            // read it (`const a: { m(): number } = { m() { return a.v; } }`, TSLANG 148)
            GenContext declarationGenContext(genContext);
            declarationGenContext.globalWrittenType = mlir::Type();
            if (!genContext.funcOp && !declarationMode && item->type && item->initializer
                && item->name == SyntaxKind::Identifier)
            {
                declarationGenContext.globalWrittenType = getType(item->type, genContext);
            }

            if (mlir::failed(processDeclaration(item, valClassItem, initFunc, declarationGenContext, true)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
    {
        auto isLet = (variableDeclarationListAST->flags & NodeFlags::Let) == NodeFlags::Let;
        auto isConst = (variableDeclarationListAST->flags & NodeFlags::Const) == NodeFlags::Const;
        auto isUsing = (variableDeclarationListAST->flags & NodeFlags::Using) == NodeFlags::Using;
        auto isExternal = (variableDeclarationListAST->flags & NodeFlags::Ambient) == NodeFlags::Ambient;
        VariableClass varClass = isExternal ? VariableType::External
                        : isLet    ? VariableType::Let
                        : isConst || isUsing ? VariableType::Const
                                   : VariableType::Var;

        varClass.isUsing = isUsing;

        auto exportByDecorator = false;
        if (variableDeclarationListAST->parent)
        {
            varClass.isPublic = hasModifier(variableDeclarationListAST->parent, SyntaxKind::ExportKeyword);
            iterateDecorators(variableDeclarationListAST->parent, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                if (name == DLL_EXPORT)
                {
                    exportByDecorator = true;
                    varClass.isExport = true;
                }

                if (name == DLL_IMPORT)
                {
                    varClass.type = isLet ? VariableType::Let : isConst || isUsing ? VariableType::Const : VariableType::Var;
                    varClass.isImport = true;
                    // it has parameter, means this is dynamic import, should point to dll path
                    // TODO: finish it, look at mlirGenCustomRTTIDynamicImport as example how to load it
                    if (args.size() > 0)
                    {
                        varClass.type = VariableType::Var;
                        varClass.isDynamicImport = true;
                        varClass.isImport = false;
                    }
                }

                if (name == "boxed") {
                    varClass.isBoxed = true;
                }

                if (name == DLL_NAME && args.size() > 0) {
                    varClass.dllName = args.front();
                }

                if (name == "used") {
                    varClass.isUsed = true;
                }

                if (name == "atomic") {
                    varClass.atomic = true;
                    if (args.size() > 0) 
                    {
                        auto ordering = 0;
                        if (llvm::to_integer(args[0], ordering))
                        {
                            varClass.ordering = ordering;
                        }
                    }

                    if (args.size() > 1)
                        varClass.syncscope = args[1];
                }

                if (name == "volatile") {
                    varClass.isVolatile = true;
                }

                if (name == "nontemporal") {
                    varClass.nonTemporal = true;
                }

                if (name == "invariant") {
                    varClass.invariant = true;
                }
            });

            if (mlir::failed(checkLinkNameDecorators(loc(variableDeclarationListAST), variableDeclarationListAST->parent, genContext)))
            {
                return mlir::failure();
            }
        }

        // An exported module-level const is a variable, as an exported `let` is. A const may be
        // folded away inside its module - a const function becomes the function and its global is
        // erased (isGlobalConstLambda) - but other modules reach it by symbol, so it has to be the
        // global they import: an importer reads `export const f = () => ...` out of variable `f`.
        for (auto &item : variableDeclarationListAST->declarations)
        {
            // we need it for support "undefined type" in 'let' without initialization
            item->parent = variableDeclarationListAST;

            // --export filters by name, and one statement can declare several names. Only a
            // variable of the module or a namespace can be exported: a filter such as `all` matched
            // every function's locals too, and declared them to importers as `@dllimport` globals
            // in a `.f_<function>` namespace, which no library defines.
            auto itemVarClass = varClass;
            // A root's statements held back into its entry function are still the module's
            // (rootStatementsFuncOp): `export const k = 5` beside top-level code was not exported,
            // and the importer of the library could not resolve k (#448).
            auto atModuleLevel = !genContext.funcOp || genContext.funcOp == genContext.rootStatementsFuncOp;
            if (variableDeclarationListAST->parent && !exportByDecorator)
            {
                itemVarClass.isExport = atModuleLevel && getExportModifier(
                    variableDeclarationListAST->parent, MLIRHelper::getName(item->name, stringAllocator));
            }

            // a module `const` beside top-level code is stored once, by the root's code where it is declared
            if (!isUsing && !itemVarClass.isImport && genContext.funcOp && genContext.rootStatementWasConst &&
                genContext.funcOp == genContext.rootStatementsFuncOp)
            {
                itemVarClass.writtenOnce = true;
            }

            // (so is one beside top-level code, a global of the module there too: GenContext::rootFunctionConst)
            if (itemVarClass.type == VariableType::Const && !isUsing && itemVarClass.isExport && !itemVarClass.isImport
                && (!genContext.funcOp || genContext.rootFunctionConst && genContext.funcOp == genContext.rootStatementsFuncOp))
            {
                itemVarClass.type = VariableType::Let;
            }

            // a `var` in a function belongs to the function (MDN `var`): declared before the body, assigned here
            if (itemVarClass.type == VariableType::Var && genContext.functionVars && !declarationMode
                && !itemVarClass.isImport && !itemVarClass.isDynamicImport)
            {
                if (mlir::failed(mlirGenFunctionVar(item, itemVarClass, genContext)))
                {
                    return mlir::failure();
                }

                continue;
            }

            // `var v = 1; var v = 2;` of the module or a namespace: one variable. Declared again it is nothing,
            // and with a value an assignment, where the module's code runs in a function of its own. (Each
            // declaration made a global of its own: "redefinition of symbol named 'v'".)
            if (itemVarClass.type == VariableType::Var && !genContext.functionVars && !declarationMode
                && !itemVarClass.isImport && !itemVarClass.isDynamicImport && item->name == SyntaxKind::Identifier)
            {
                auto fullName = concatFullNamespaceName(MLIRHelper::getName(item->name));
                auto [first, inserted] = moduleVarFirstDeclaration.try_emplace(fullName, item->pos.pos);
                if (!inserted && first->second != item->pos.pos)
                {
                    if (!item->initializer)
                    {
                        continue;
                    }

                    // outside a function (a namespace's or an imported module's declarations, or a dummy run): the
                    // assignment is the code's, which runs it (generateModuleInitCode, isRepeatedVarStatement). It was
                    // a second global: "redefinition of symbol named 'C.p'" (TSLANG 174)
                    if (!genContext.funcOp)
                    {
                        continue;
                    }

                    {

                        NodeFactory nf(NodeFactoryFlags::None);
                        auto identifier = nf.createIdentifier(stows(MLIRHelper::getName(item->name)));
                        auto assignment = nf.createBinaryExpression(identifier, nf.createToken(SyntaxKind::EqualsToken),
                                                                    item->initializer);
                        for (Node node : {Node(identifier), Node(assignment)})
                        {
                            node->pos = item->pos;
                            node->_end = item->_end;
                        }

                        auto result = mlirGen(assignment.as<Expression>(), genContext);
                        EXIT_IF_FAILED(result)
                        continue;
                    }
                }
            }

            // ...and a destructuring `var` all of whose names were declared before
            // (`var { name: a } = r; var { name: a } = s;`): an assignment to them
            if (itemVarClass.type == VariableType::Var && !genContext.functionVars && !declarationMode
                && !itemVarClass.isImport && !itemVarClass.isDynamicImport && item->initializer
                && (item->name == SyntaxKind::ObjectBindingPattern || item->name == SyntaxKind::ArrayBindingPattern))
            {
                SmallVector<std::string> names;
                collectBindingPatternNames(item->name, names);
                auto allDeclared = !names.empty() && llvm::all_of(names, [&](const std::string &name) {
                    auto found = moduleVarFirstDeclaration.find(concatFullNamespaceName(name));
                    return found != moduleVarFirstDeclaration.end() && found->second != item->pos.pos;
                });

                if (allDeclared)
                {
                    if (!genContext.funcOp && genContext.dummyRun)
                    {
                        // a dummy run of the module's statements: the assignment is the real run's
                        continue;
                    }

                    NodeFactory nf(NodeFactoryFlags::None);
                    auto target = bindingPatternAsAssignmentTarget(nf, item->name);
                    auto assignment = nf.createBinaryExpression(target, nf.createToken(SyntaxKind::EqualsToken),
                                                                item->initializer);
                    assignment->pos = item->pos;
                    assignment->_end = item->_end;

                    if (genContext.funcOp)
                    {
                        auto result = mlirGen(assignment.as<Expression>(), genContext);
                        EXIT_IF_FAILED(result)
                        continue;
                    }

                    // no code of the module runs in a function: the assignment runs from a constructor of its own,
                    // as a destructuring declaration there does (mlirGen(VariableDeclaration))
                    auto location = loc(item);
                    auto name = MLIRHelper::getAnonymousName(location, ".gc", "");
                    auto fullInitGlobalFuncName = getFullNamespaceName(name);
                    {
                        mlir::OpBuilder::InsertionGuard insertGuard(builder);
                        if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, getFunctionType({}, {}, false),
                            [&](mlir::Location location, const GenContext &genContext) {
                                auto result = mlirGen(assignment.as<Expression>(), genContext);
                                EXIT_IF_FAILED(result)
                                return mlir::success();
                            }, genContext)))
                        {
                            return mlir::failure();
                        }

                        addGlobalConstructor(location, fullInitGlobalFuncName, /*afterGlobals=*/true);
                    }

                    continue;
                }

                for (auto &name : names)
                {
                    moduleVarFirstDeclaration.try_emplace(concatFullNamespaceName(name), item->pos.pos);
                }
            }

            if (mlir::failed(mlirGen(item, itemVarClass, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    // the names a destructuring pattern declares
    void MLIRGenImpl::collectBindingPatternNames(Node name, SmallVector<std::string> &names)
    {
        if (name == SyntaxKind::Identifier)
        {
            names.push_back(MLIRHelper::getName(name.as<Identifier>()));
            return;
        }

        if (name == SyntaxKind::ObjectBindingPattern)
        {
            for (auto element : name.as<ObjectBindingPattern>()->elements)
            {
                collectBindingPatternNames(element->name, names);
            }
        }
        else if (name == SyntaxKind::ArrayBindingPattern)
        {
            for (auto element : name.as<ArrayBindingPattern>()->elements)
            {
                if (element == SyntaxKind::BindingElement)
                {
                    collectBindingPatternNames(element.as<BindingElement>()->name, names);
                }
            }
        }
    }

    // `{ a: x, y = 1, ...r }` / `[x, , y = 1, ...r]` as the target of an assignment, as written in one
    Expression MLIRGenImpl::bindingPatternAsAssignmentTarget(NodeFactory &nf, Node name)
    {
        // each node made here is where the one it is made from is: a node with no place in the source crashed
        // what reports or locates it
        auto place = [](auto node, Node from) {
            node->pos = from->pos;
            node->_end = from->_end;
            return node;
        };

        if (name == SyntaxKind::Identifier)
        {
            return place(nf.createIdentifier(stows(MLIRHelper::getName(name.as<Identifier>()))), name);
        }

        auto withDefault = [&](Expression target, Expression initializer, Node from) -> Expression {
            if (!initializer)
            {
                return target;
            }

            return place(nf.createBinaryExpression(target, place(nf.createToken(SyntaxKind::EqualsToken), from), initializer),
                         from).template as<Expression>();
        };

        if (name == SyntaxKind::ObjectBindingPattern)
        {
            NodeArray<ObjectLiteralElementLike> properties;
            for (auto element : name.as<ObjectBindingPattern>()->elements)
            {
                if (element->dotDotDotToken)
                {
                    properties.push_back(
                        place(nf.createSpreadAssignment(bindingPatternAsAssignmentTarget(nf, element->name)), element));
                }
                else if (!element->propertyName && element->name == SyntaxKind::Identifier)
                {
                    properties.push_back(place(nf.createShorthandPropertyAssignment(
                        place(nf.createIdentifier(stows(MLIRHelper::getName(element->name.as<Identifier>()))), element->name),
                        element->initializer), element));
                }
                else
                {
                    auto propertyName = element->propertyName ? element->propertyName : element->name.as<PropertyName>();
                    properties.push_back(place(nf.createPropertyAssignment(
                        propertyName,
                        withDefault(bindingPatternAsAssignmentTarget(nf, element->name), element->initializer, element)),
                        element));
                }
            }

            return place(nf.createObjectLiteralExpression(properties, false), name);
        }

        NodeArray<Expression> elements;
        for (auto element : name.as<ArrayBindingPattern>()->elements)
        {
            if (element != SyntaxKind::BindingElement)
            {
                elements.push_back(place(nf.createOmittedExpression(), element));
                continue;
            }

            auto bindingElement = element.as<BindingElement>();
            auto target = bindingPatternAsAssignmentTarget(nf, bindingElement->name);
            if (bindingElement->dotDotDotToken)
            {
                elements.push_back(place(nf.createSpreadElement(target), element).template as<Expression>());
            }
            else
            {
                elements.push_back(withDefault(target, bindingElement->initializer, element));
            }
        }

        return place(nf.createArrayLiteralExpression(elements, false), name);
    }

    // The `var` declarations of a function body, in source order: in its blocks, branches, loops, `try`, `switch` and
    // labels, not in the functions and classes it declares, which have their own. A `for...in`/`for...of` header's
    // `var` is marked: its type comes from what the loop iterates (`inLoopHeader`, kept out of the early pass).
    void MLIRGenImpl::collectFunctionVarDeclarations(Node node, SmallVector<VariableDeclaration> &declarations,
                                                     bool inLoopHeader)
    {
        if (!node)
        {
            return;
        }

        auto collectList = [&](VariableDeclarationList list, bool header) {
            if ((list->flags & (NodeFlags::Let | NodeFlags::Const | NodeFlags::Using | NodeFlags::Ambient)) != NodeFlags::None)
            {
                return;
            }

            if (header)
            {
                return;
            }

            for (auto declaration : list->declarations)
            {
                declarations.push_back(declaration);
            }
        };

        switch ((SyntaxKind)node)
        {
        case SyntaxKind::Block:
            for (auto statement : node.as<ts::Block>()->statements)
            {
                collectFunctionVarDeclarations(statement, declarations);
            }

            break;
        case SyntaxKind::VariableStatement:
            collectList(node.as<VariableStatement>()->declarationList, false);
            break;
        case SyntaxKind::IfStatement:
            collectFunctionVarDeclarations(node.as<IfStatement>()->thenStatement, declarations);
            collectFunctionVarDeclarations(node.as<IfStatement>()->elseStatement, declarations);
            break;
        case SyntaxKind::ForStatement:
        {
            auto forStatement = node.as<ForStatement>();
            if (forStatement->initializer == SyntaxKind::VariableDeclarationList)
            {
                collectList(forStatement->initializer.as<VariableDeclarationList>(), false);
            }

            collectFunctionVarDeclarations(forStatement->statement, declarations);
            break;
        }
        case SyntaxKind::ForInStatement:
            collectFunctionVarDeclarations(node.as<ForInStatement>()->statement, declarations);
            break;
        case SyntaxKind::ForOfStatement:
            collectFunctionVarDeclarations(node.as<ForOfStatement>()->statement, declarations);
            break;
        case SyntaxKind::WhileStatement:
            collectFunctionVarDeclarations(node.as<WhileStatement>()->statement, declarations);
            break;
        case SyntaxKind::DoStatement:
            collectFunctionVarDeclarations(node.as<DoStatement>()->statement, declarations);
            break;
        case SyntaxKind::LabeledStatement:
            collectFunctionVarDeclarations(node.as<LabeledStatement>()->statement, declarations);
            break;
        case SyntaxKind::TryStatement:
        {
            auto tryStatement = node.as<TryStatement>();
            collectFunctionVarDeclarations(tryStatement->tryBlock, declarations);
            if (tryStatement->catchClause)
            {
                collectFunctionVarDeclarations(tryStatement->catchClause->block, declarations);
            }

            collectFunctionVarDeclarations(tryStatement->finallyBlock, declarations);
            break;
        }
        case SyntaxKind::SwitchStatement:
            for (auto clause : node.as<SwitchStatement>()->caseBlock->clauses)
            {
                for (auto statement : clause->statements)
                {
                    collectFunctionVarDeclarations(statement, declarations);
                }
            }

            break;
        default:
            break;
        }
    }

    // Declares the `var` named `name` as a variable of the function (a `let` with no value yet): in the function's
    // own symbol scope, its storage after the function's prologue, owned by the function's frame, so the blocks of
    // the function all see one variable that lives as long as the call.
    mlir::LogicalResult MLIRGenImpl::declareFunctionVar(mlir::Location location, StringRef name, mlir::Type type,
                                                        bool typeProvided, const GenContext &genContext)
    {
        GenContext varGenContext(*genContext.functionVarContext);
        varGenContext.declareIntoScope = genContext.functionVarScope;

        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        if (genContext.functionVarAnchor)
        {
            builder.setInsertionPointAfter(genContext.functionVarAnchor);
        }
        else
        {
            builder.setInsertionPointToStart(genContext.functionVarBlock);
        }

        genContext.functionVars->insert(name);

        auto declaredType = registerVariable(
            location, name, false, VariableType::Let,
            [&](mlir::Location, const GenContext &) {
                return std::make_tuple(type, mlir::Value(), typeProvided ? TypeProvided::Yes : TypeProvided::No);
            },
            varGenContext);

        return declaredType ? mlir::success() : mlir::failure();
    }

    // Before a function's body: each `var` the body declares whose type can be known there (written, or of an
    // initializer that reads what is already declared: parameters, the `var`s before it) is declared, so it can be
    // read and written anywhere in the function, before its declaration too. A `var` with neither type nor
    // initializer anywhere is an `any`. The others are declared where they are first reached (mlirGenFunctionVar).
    // A `var` typed by a generic function value (`var f = x => x`) holds that function instantiated with
    // `any` (instantiateGenericFunctionWithAny): its type is that instance's, found without making it. The
    // assignment that gives the var its value instantiates the function against this type.
    mlir::Type MLIRGenImpl::functionVarTypeOfGenericInit(VariableDeclaration typing, mlir::Type type,
                                                         const GenContext &genContext)
    {
        // `var k = C`: the class's value; the evaluated type alone is that of an instance
        if (isa_and_nonnull<mlir_ts::ClassType>(type) && typing->initializer)
        {
            mlir::Type classValueType;
            evaluate(
                typing->initializer,
                [&](mlir::Value value) {
                    if (auto classRefOp = value.getDefiningOp<mlir_ts::ClassRefOp>(); classRefOp && !getValueClassInfo(classRefOp.getType()))
                    {
                        classValueType = mlir_ts::ClassValueType::get(classRefOp.getType());
                    }
                },
                genContext);
            return classValueType ? classValueType : type;
        }

        auto funcType = dyn_cast_or_null<mlir_ts::FunctionType>(type);
        if (!funcType || !mth.isGenericType(funcType) || !typing->initializer)
        {
            return type;
        }

        mlir::Type instanceType;
        evaluate(
            typing->initializer,
            [&](mlir::Value value) {
                if (!isGenericFunctionReference(value))
                {
                    return;
                }

                if (auto receiverType = anyReceiverOfGenericFunction(value))
                {
                    instanceType = instantiateSpecializedFunctionTypeHelper(loc(typing), value, receiverType,
                                                                            true, genContext);
                }
            },
            genContext);
        return instanceType;
    }

    mlir::LogicalResult MLIRGenImpl::declareFunctionVars(Node body, const GenContext &genContext)
    {
        // A `let`/`const` written directly in the body that a function before it reads (`const f = () => n; const n
        // = 1;`): the function is made before the declaration is reached, and could not capture what was no name
        // yet ("can't resolve name: n"). It is one of the function's variables, declared before the body, as a
        // `var` is; its declaration gives it its value.
        llvm::StringSet<> forwardReferenced;
        if (body == SyntaxKind::Block)
        {
            auto &statements = body.as<ts::Block>()->statements;
            for (auto [index, statement] : enumerate(statements))
            {
                if (statement != SyntaxKind::VariableStatement)
                {
                    continue;
                }

                auto declarationList = statement.as<VariableStatement>()->declarationList;
                auto collectNames = [&]() {
                    for (auto declaration : declarationList->declarations)
                    {
                        if (declaration->name == SyntaxKind::Identifier)
                        {
                            forwardReferenced.insert(MLIRHelper::getName(declaration->name));
                        }
                    }
                };

                // made one of the function's variables by an earlier pass over this body
                if ((statement->internalFlags & InternalFlags::HoistedLexical) == InternalFlags::HoistedLexical)
                {
                    collectNames();
                    continue;
                }

                // a `var` used before its declaration (`x = 7; var x = one();`): declared before the body, as a `var`
                // is, with the type its initializer gives; it was "can't resolve name: x" (TSLANG 172). Not under
                // -mm=own, which refuses a variable declared without a value that is given one later.
                if ((declarationList->flags & (NodeFlags::Let | NodeFlags::Const | NodeFlags::Using)) == NodeFlags::None)
                {
                    if (compileOptions.memoryModel == MemoryModelOwn)
                    {
                        continue;
                    }

                    for (auto declaration : declarationList->declarations)
                    {
                        if (declaration->name != SyntaxKind::Identifier || !declaration->initializer)
                        {
                            continue;
                        }

                        auto name = declaration->name.as<Identifier>()->escapedText;
                        auto usedBefore = false;
                        for (size_t earlier = 0; earlier < index && !usedBefore; earlier++)
                        {
                            ts::VisitorAST names([&](ts::Node inner) {
                                if (inner == SyntaxKind::Identifier && inner.as<Identifier>()->escapedText == name)
                                {
                                    usedBefore = true;
                                }
                            });
                            names.visit(statements[earlier]);
                        }

                        if (usedBefore)
                        {
                            forwardReferenced.insert(MLIRHelper::getName(declaration->name));
                        }
                    }

                    continue;
                }

                if ((declarationList->flags & NodeFlags::Using) == NodeFlags::Using)
                {
                    continue;
                }

                auto readBefore = false;
                for (auto declaration : declarationList->declarations)
                {
                    if (declaration->name != SyntaxKind::Identifier || !declaration->initializer)
                    {
                        continue;
                    }

                    auto name = declaration->name.as<Identifier>()->escapedText;
                    for (size_t earlier = 0; earlier < index && !readBefore; earlier++)
                    {
                        ts::VisitorAST functions([&](ts::Node node) {
                            if (readBefore
                                || !(node == SyntaxKind::ArrowFunction || node == SyntaxKind::FunctionExpression
                                     || node == SyntaxKind::FunctionDeclaration || node == SyntaxKind::MethodDeclaration
                                     || node == SyntaxKind::GetAccessor || node == SyntaxKind::SetAccessor))
                            {
                                return;
                            }

                            ts::VisitorAST names([&](ts::Node inner) {
                                if (inner == SyntaxKind::Identifier && inner.as<Identifier>()->escapedText == name)
                                {
                                    readBefore = true;
                                }
                            });
                            names.visit(node);
                        });
                        functions.visit(statements[earlier]);
                    }
                }

                if (readBefore)
                {
                    declarationList->flags &= ~(NodeFlags::Let | NodeFlags::Const);
                    statement->internalFlags |= InternalFlags::HoistedLexical;
                    collectNames();
                }
            }

            // what a hoisted declaration's value reads is declared before it too: its type is that value's
            // (`const f = () => g(); const n = 5; const g = () => n;` - g's type needs n; it was "can't resolve name:
            // g", TSLANG 174)
            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto statement : statements)
                {
                    if (statement != SyntaxKind::VariableStatement
                        || (statement->internalFlags & InternalFlags::HoistedLexical) == InternalFlags::HoistedLexical)
                    {
                        continue;
                    }

                    auto declarationList = statement.as<VariableStatement>()->declarationList;
                    if ((declarationList->flags & (NodeFlags::Let | NodeFlags::Const)) == NodeFlags::None
                        || (declarationList->flags & NodeFlags::Using) == NodeFlags::Using)
                    {
                        continue;
                    }

                    auto readByHoisted = false;
                    for (auto declaration : declarationList->declarations)
                    {
                        if (declaration->name != SyntaxKind::Identifier || !declaration->initializer)
                        {
                            continue;
                        }

                        auto name = declaration->name.as<Identifier>()->escapedText;
                        for (auto hoisted : statements)
                        {
                            if (readByHoisted || hoisted == statement || hoisted != SyntaxKind::VariableStatement
                                || (hoisted->internalFlags & InternalFlags::HoistedLexical) != InternalFlags::HoistedLexical)
                            {
                                continue;
                            }

                            ts::VisitorAST names([&](ts::Node inner) {
                                if (inner == SyntaxKind::Identifier && inner.as<Identifier>()->escapedText == name)
                                {
                                    readByHoisted = true;
                                }
                            });
                            // the value of a function: its type is the function's, read off what it reads
                            for (auto hoistedDeclaration : hoisted.as<VariableStatement>()->declarationList->declarations)
                            {
                                if (hoistedDeclaration->initializer == SyntaxKind::ArrowFunction
                                    || hoistedDeclaration->initializer == SyntaxKind::FunctionExpression)
                                {
                                    names.visit(hoistedDeclaration->initializer);
                                }
                            }
                        }
                    }

                    if (readByHoisted)
                    {
                        declarationList->flags &= ~(NodeFlags::Let | NodeFlags::Const);
                        statement->internalFlags |= InternalFlags::HoistedLexical;
                        for (auto declaration : declarationList->declarations)
                        {
                            if (declaration->name == SyntaxKind::Identifier)
                            {
                                forwardReferenced.insert(MLIRHelper::getName(declaration->name));
                            }
                        }

                        changed = true;
                    }
                }
            }
        }

        SmallVector<VariableDeclaration> declarations;
        collectFunctionVarDeclarations(body, declarations);

        // the declarations written directly in the body, not in a block of it
        auto &topLevel = *genContext.functionTopLevelVars;
        if (body == SyntaxKind::Block)
        {
            for (auto statement : body.as<ts::Block>()->statements)
            {
                if (statement == SyntaxKind::VariableStatement)
                {
                    for (auto declaration : statement.as<VariableStatement>()->declarationList->declarations)
                    {
                        topLevel.insert(declaration->pos.pos);
                    }
                }
            }
        }

        llvm::StringSet<> deferred;
        for (auto declaration : declarations)
        {
            if (declaration->name != SyntaxKind::Identifier)
            {
                continue;
            }

            auto name = MLIRHelper::getName(declaration->name, stringAllocator);
            if (genContext.functionVars->contains(name) || deferred.contains(name))
            {
                continue;
            }

            // first declared, with its value, directly in the body: declared there, with the value, as `let` is
            // (-mm=own refuses a variable declared without a value that is given one later)
            if (declaration->initializer && topLevel.contains(declaration->pos.pos) && !forwardReferenced.contains(name))
            {
                deferred.insert(name);
                continue;
            }

            // the declaration that gives the type: the first one with a type, else the first with an initializer
            VariableDeclaration typing;
            auto anyTyped = false;
            auto anyInitialized = false;
            for (auto other : declarations)
            {
                if (other->name != SyntaxKind::Identifier || MLIRHelper::getName(other->name) != name.str())
                {
                    continue;
                }

                if (other->type && !anyTyped)
                {
                    typing = other;
                    anyTyped = true;
                }
                else if (other->initializer && !anyTyped && !anyInitialized)
                {
                    typing = other;
                    anyInitialized = true;
                }
            }

            mlir::Type type;
            if (!typing)
            {
                type = getAnyType();
            }
            else
            {
                auto savedErrorMessagesCount = genContext.postponedMessages ? genContext.postponedMessages->size() : 0;
                auto [evaluatedType, hasInit, typeProvided] = evaluateTypeAndInit(typing, genContext);
                if (!typeProvided)
                {
                    evaluatedType = functionVarTypeOfGenericInit(typing, evaluatedType, genContext);
                }

                if (genContext.postponedMessages)
                {
                    rollbackPostponedErrorMessages(genContext.postponedMessages, savedErrorMessagesCount);
                }

                type = evaluatedType;
                if (!type)
                {
                    // known only where it is reached
                    deferred.insert(name);
                    continue;
                }

                if (!typeProvided)
                {
                    // as `let x = 1` holds any number
                    type = mth.wideStorageType(mth.stripLiteralType(type));
                }
            }

            if (mlir::failed(declareFunctionVar(loc(declaration), name, type, anyTyped, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    // `var x = e` in a function: an assignment to the function's `x`, declared before the body or here, the first time
    // it is reached. `var x` again is nothing. A destructuring `var` is a variable of the block it is in, as `let`.
    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionVar(VariableDeclaration item, VariableClass varClass,
                                                        const GenContext &genContext)
    {
        auto location = loc(item);
        if (item->name != SyntaxKind::Identifier)
        {
            VariableClass letClass = varClass;
            letClass.type = VariableType::Let;
            auto topLevel = genContext.functionTopLevelVars && genContext.functionTopLevelVars->contains(item->pos.pos);
            if (topLevel || !item->initializer || compileOptions.memoryModel == MemoryModelOwn)
            {
                return mlirGen(item, letClass, genContext);
            }

            // `{ var [a, b] = t; } return a + b;`: the names are the function's, read after the block too ("can't
            // resolve name: a"). The pattern is taken apart into `let`s of the block named `.var.a`, ...; each name
            // is then one of the function's variables, of that `let`'s type, given its value.
            SmallVector<Identifier> names;
            // `{ x }`: the property is named by the name, which is renamed; it is named for the call
            SmallVector<BindingElement> shorthands;
            NodeFactory nf(NodeFactoryFlags::None);
            std::function<void(Node)> collect = [&](Node pattern) {
                SmallVector<Node> elements;
                if (pattern == SyntaxKind::ObjectBindingPattern)
                {
                    for (auto element : pattern.as<ObjectBindingPattern>()->elements)
                    {
                        elements.push_back(element);
                    }
                }
                else
                {
                    for (auto element : pattern.as<ArrayBindingPattern>()->elements)
                    {
                        elements.push_back(element);
                    }
                }

                for (auto element : elements)
                {
                    if (element != SyntaxKind::BindingElement)
                    {
                        continue;
                    }

                    auto bindingElement = element.as<BindingElement>();
                    auto elementName = bindingElement->name;
                    if (elementName == SyntaxKind::Identifier)
                    {
                        names.push_back(elementName.as<Identifier>());
                        if (pattern == SyntaxKind::ObjectBindingPattern && !bindingElement->propertyName)
                        {
                            auto propertyName = nf.createIdentifier(stows(MLIRHelper::getName(elementName)));
                            propertyName->pos = elementName->pos;
                            propertyName->_end = elementName->_end;
                            bindingElement->propertyName = propertyName;
                            shorthands.push_back(bindingElement);
                        }
                    }
                    else
                    {
                        collect(elementName);
                    }
                }
            };
            collect(item->name);

            SmallVector<string> written;
            for (auto name : names)
            {
                written.push_back(name->escapedText);
                name->escapedText = S(".var.") + stows(MLIRHelper::getName(name));
            }

            auto result = mlirGen(item, letClass, genContext);
            SmallVector<string> innerNames;
            for (auto [name, text] : llvm::zip(names, written))
            {
                innerNames.push_back(name->escapedText);
                name->escapedText = text;
            }

            for (auto shorthand : shorthands)
            {
                shorthand->propertyName = undefined;
            }

            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            for (auto [name, innerName] : llvm::zip(names, innerNames))
            {
                auto plainName = MLIRHelper::getName(name, stringAllocator);
                auto inner = nf.createIdentifier(innerName);
                inner->pos = name->pos;
                inner->_end = name->_end;
                if (!genContext.functionVars->contains(plainName))
                {
                    auto innerValue = mlirGen(inner.as<Expression>(), genContext);
                    EXIT_IF_FAILED_OR_NO_VALUE(innerValue)
                    auto type = mth.wideStorageType(mth.stripLiteralType(V(innerValue).getType()));
                    if (mlir::failed(declareFunctionVar(loc(name), plainName, type, false, genContext)))
                    {
                        return mlir::failure();
                    }
                }

                auto outer = nf.createIdentifier(stows(plainName.str()));
                auto assignment = nf.createBinaryExpression(outer, nf.createToken(SyntaxKind::EqualsToken), inner);
                for (Node node : {Node(outer), Node(assignment)})
                {
                    node->pos = name->pos;
                    node->_end = name->_end;
                }

                auto assigned = mlirGen(assignment.as<Expression>(), genContext);
                EXIT_IF_FAILED(assigned)
            }

            return mlir::success();
        }

        auto name = MLIRHelper::getName(item->name, stringAllocator);
        // first reached directly in the body, with its value: a `let` of the function's own scope, declared with it
        if (!genContext.functionVars->contains(name) && item->initializer && genContext.functionTopLevelVars
            && genContext.functionTopLevelVars->contains(item->pos.pos))
        {
            VariableClass letClass = varClass;
            letClass.type = VariableType::Let;
            if (mlir::failed(mlirGen(item, letClass, genContext)))
            {
                return mlir::failure();
            }

            genContext.functionVars->insert(name);
            return mlir::success();
        }

        if (!genContext.functionVars->contains(name))
        {
            auto [type, hasInit, typeProvided] = evaluateTypeAndInit(item, genContext);
            if (!type && !hasInit)
            {
                type = getAnyType();
            }

            if (type && !typeProvided)
            {
                type = functionVarTypeOfGenericInit(item, type, genContext);
            }

            if (!type)
            {
                // the initializer's own error
                auto result = mlirGen(item->initializer, genContext);
                EXIT_IF_FAILED(result)
                emitError(location) << "Can't resolve variable '" << name << "' type";
                return mlir::failure();
            }

            if (!typeProvided)
            {
                type = mth.wideStorageType(mth.stripLiteralType(type));
            }

            if (mlir::failed(declareFunctionVar(location, name, type, typeProvided, genContext)))
            {
                return mlir::failure();
            }
        }

        if (!item->initializer)
        {
            return mlir::success();
        }

        NodeFactory nf(NodeFactoryFlags::None);
        auto identifier = nf.createIdentifier(stows(name.str()));
        auto assignment =
            nf.createBinaryExpression(identifier, nf.createToken(SyntaxKind::EqualsToken), item->initializer);
        for (Node node : {Node(identifier), Node(assignment)})
        {
            node->pos = item->pos;
            node->_end = item->_end;
        }

        auto result = mlirGen(assignment.as<Expression>(), genContext);
        EXIT_IF_FAILED(result)
        return mlir::success();
    }

    mlir::Type MLIRGenImpl::mlirGenParameterObjectOrArrayBinding(Node name, const GenContext &genContext)
    {
        // TODO: put it into function to support recursive call
        if (name == SyntaxKind::ObjectBindingPattern)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;

            // we need to construct object type
            auto objectBindingPattern = name.as<ObjectBindingPattern>();
            for (auto objectBindingElement : objectBindingPattern->elements)
            {
                mlirGenParameterBindingElement(objectBindingElement, fieldInfos, genContext);
            }

            return getTupleType(fieldInfos);
        } 
        else if (name == SyntaxKind::ArrayBindingPattern)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;

            // we need to construct object type
            // an element nothing gives a type (`function f([x, , z]) {}`) is `any`, as TypeScript's implicit any:
            // "can't resolve type for binding pattern 'x', provide default initializer". A skipped one holds its
            // place, so the ones after it are read at theirs.
            // (an element of no name, as `[any, any]`'s are: an array literal given is taken as such a tuple)
            auto anyElement = [&]() {
                fieldInfos.push_back({mlir::Attribute(), getAnyType(), false, mlir_ts::AccessLevel::Public});
            };

            auto arrayBindingPattern = name.as<ArrayBindingPattern>();
            for (auto arrayBindingElement : arrayBindingPattern->elements)
            {
                if (arrayBindingElement == SyntaxKind::OmittedExpression)
                {
                    anyElement();
                    continue;
                }

                if (arrayBindingElement == SyntaxKind::BindingElement)
                {
                    auto objectBindingElement = arrayBindingElement.as<BindingElement>();
                    if (!objectBindingElement->initializer && !objectBindingElement->dotDotDotToken
                        && objectBindingElement->name == SyntaxKind::Identifier)
                    {
                        anyElement();
                        continue;
                    }

                    mlirGenParameterBindingElement(objectBindingElement, fieldInfos, genContext);
                }
            }

            return getTupleType(fieldInfos);
        }        

        return mlir::Type();
    }

    mlir::Value MLIRGenImpl::resolveIdentifierAsVariable(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (name.empty())
        {
            return mlir::Value();
        }

        auto value = symbolTable.lookup(name);
        if (value.second && value.first)
        {
            //LLVM_DEBUG(dbgs() << "\n!! resolveIdentifierAsVariable: " << name << " type: " << value.second->getType() <<  " value: " << value.first;);

            auto valueRegion = value.first.getParentRegion();

            // a local of a function this one is not declared in - the caller a generic is instantiated
            // from - is no name here: the name is the module's (lexicalFuncOps, TSLANG_ISSUES 30)
            if (!lexicalFuncOps.empty() && valueRegion && valueRegion->getParentOp() &&
                (isa<mlir_ts::FuncOp>(valueRegion->getParentOp()) || valueRegion->getParentOp()->getParentOfType<mlir_ts::FuncOp>()))
            {
                auto inRoot = llvm::any_of(lexicalFuncOps, [&](mlir_ts::FuncOp lexicalFuncOp) {
                    return lexicalFuncOp.getCallableRegion() && lexicalFuncOp.getCallableRegion()->isAncestor(valueRegion);
                });
                mlir_ts::FuncOp contextFuncOp = genContext.funcOp;
                auto inFunction = contextFuncOp && contextFuncOp.getCallableRegion() &&
                                  contextFuncOp.getCallableRegion()->isAncestor(valueRegion);
                // a module's top-level code is a function of its own (rootStatementsFuncOp), whose
                // locals are the module's variables
                mlir_ts::FuncOp rootStatementsFuncOp = genContext.rootStatementsFuncOp;
                auto inModuleCode = rootStatementsFuncOp && rootStatementsFuncOp.getCallableRegion() &&
                                    rootStatementsFuncOp.getCallableRegion()->isAncestor(valueRegion);
                if (!inRoot && !inFunction && !inModuleCode)
                {
                    return mlir::Value();
                }
            }

            // begin of logic: outer vars
            auto isOuterVar = false;
            // TODO: review code "valueRegion && valueRegion->getParentOp()" is to support async.execute
            if (genContext.funcOp && genContext.funcOp != tempFuncOp && valueRegion &&
                valueRegion->getParentOp() /* && valueRegion->getParentOp()->getParentOp()*/)
            {
                mlir_ts::FuncOp contextFuncOp = genContext.funcOp;
                auto funcRegion = contextFuncOp.getCallableRegion();

                isOuterVar = !funcRegion->isAncestor(valueRegion);
                // TODO: HACK
                if (isOuterVar && value.second->getIgnoreCapturing())
                {
                    // special case when "ForceConstRef" pointering to outer variable but it is not outer var
                    isOuterVar = false;
                }

                LLVM_DEBUG(if (isOuterVar) dbgs() << "\n!! outer var: [" << value.second->getName()
                                  << "] \n\n\tvalue region: " << *valueRegion->getParentOp()
                                  << " \n\n\tFuncOp: " << contextFuncOp << "";);
            }

            if (isOuterVar && genContext.passResult && !isGenericFunctionReference(value.first))
            {
                LLVM_DEBUG(dbgs() << "\n!! capturing var: [" << value.second->getName()
                                  << "] \n\tvalue pair: " << value.first << " \n\ttype: " << value.second->getType()
                                  << " \n\treadwrite: " << value.second->getReadWriteAccess() << "";);

                // debug ref of ref
                assert(!isa<mlir_ts::RefType>(value.second->getType()));

                // valueRegion->viewGraph();

                // special case, to prevent capturing ".a" because of reference to outer VaribleOp, which is hack (review
                // solution for it)
                genContext.passResult->outerVariables.insert({value.second->getName(), value.second});
                genContext.passResult->outerVariableUses.insert({value.second->getName(), mlir::LocationAttr(location)});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            //LLVM_DEBUG(dbgs() << "\n!! variable: " << name << " type: " << value.first.getType() << "\n");

            // load value if memref
            auto valueType = mlir::cast<mlir_ts::RefType>(value.first.getType()).getElementType();
            auto loadOp = builder.create<mlir_ts::LoadOp>(location, valueType, value.first);
            if (value.second->getAtomic())
            {
                loadOp->setAttr(ATOMIC_ATTR_NAME, builder.getBoolAttr(true));
                loadOp->setAttr(ORDERING_ATTR_NAME, builder.getI32IntegerAttr(value.second->getOrdering()));
                loadOp->setAttr(SYNCSCOPE_ATTR_NAME, builder.getStringAttr(value.second->getSyncScope()));
            }

            if (value.second->getVolatile())
            {
                loadOp->setAttr(VOLATILE_ATTR_NAME, builder.getBoolAttr(true));
            }

            if (value.second->getNonTemporal())
            {
                loadOp->setAttr(NONTEMPORAL_ATTR_NAME, builder.getBoolAttr(true));
            }            

            if (value.second->getInvariant())
            {
                loadOp->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
            }            

            // a generator's narrowed local: what the local holds now, cast as the test narrowed it
            if (auto narrowedType = value.second->getNarrowedType())
            {
                auto castValue = castToNarrowedType(location, loadOp, narrowedType, genContext);
                if (!castValue)
                {
                    return mlir::Value();
                }

                return builder.create<mlir_ts::SafeCastOp>(location, narrowedType, castValue, loadOp);
            }

            return loadOp;
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveFunctionNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto &funcEntry = fn->getValue();
            return resolveFunctionWithCapture(location, funcEntry.name, funcEntry.funcType, mlir::Value(), false, genContext);
        }

        return mlir::Value();
    }

    // `type Node = { value: number, next: Node }`: the declaration names itself, and each mention
    // expanded it again until the stack ran out. A type here has a fixed shape - a tuple holds its
    // fields in place - so no alias can contain itself, not even behind `Reference<T>`, whose tuple
    // type would have to name itself; an interface or a class is the recursive type. A generic
    // alias is the same: `List<T>` inside `List<T>` expands without end, whatever its arguments.
    //
    // One error says it. Every error on the way back out of the expansion follows from it, and so
    // does "can't find type" at each later mention of the alias; those are left out.
    mlir::Type MLIRGenImpl::expandTypeAlias(mlir::Location location, StringRef name, llvm::function_ref<mlir::Type()> expand)
    {
        auto fullName = getFullNamespaceName(name).str();
        if (!typeAliasesInProgress.insert(fullName).second)
        {
            reportingCircularTypeAlias = true;
            emitError(location) << "type alias '" << name
                                << "' circularly references itself; use an interface or a class for a recursive type";
            reportingCircularTypeAlias = false;
            circularTypeAliases.insert(name);
            circularTypeAliasUnwinding = fullName;
            return mlir::Type();
        }

        mlir::Type type;
        {
            mlir::ScopedDiagnosticHandler followOn(builder.getContext(), [&](mlir::Diagnostic &) {
                return mlir::success(!circularTypeAliasUnwinding.empty() && !reportingCircularTypeAlias);
            });

            type = expand();
        }

        typeAliasesInProgress.erase(fullName);
        if (circularTypeAliasUnwinding == fullName)
        {
            circularTypeAliasUnwinding.clear();
        }

        return type;
    }

    mlir::Type MLIRGenImpl::resolveTypeByNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // support generic types
        if (genContext.typeParamsWithArgs.size() > 0)
        {
            auto type = getResolveTypeParameter(name, false, genContext);
            if (type)
            {
                return type;
            }
        }

        if (genContext.typeAliasMap.count(name))
        {
            auto typeAliasInfo = genContext.typeAliasMap.lookup(name);
            assert(typeAliasInfo);
            return typeAliasInfo;
        }

        if (getTypeAliasMap().count(name))
        {
            auto typeAliasInfo = getTypeAliasMap().lookup(name);
            if (typeAliasInfo.first)
            {
                return typeAliasInfo.first;
            }

            assert(typeAliasInfo.second);
            return expandTypeAlias(location, name, [&]() {
                GenContext typeAliasGenContext(genContext);
                return getType(typeAliasInfo.second, typeAliasGenContext);
            });
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            if (!classInfo->classType)
            {
                emitError(location) << "can't find class: " << name << "\n";
                return mlir::Type();
            }

            // a @value class is a value of its fields; used as a value (`V.plus`), its name is still
            // the class (resolveIdentifier), whose statics are as any class's
            if (classInfo->isValue)
            {
                return getValueTypeOfClassIfLaidOut(classInfo);
            }

            return classInfo->classType;
        }

        if (getGenericClassesMap().count(name))
        {
            auto genericClassInfo = getGenericClassesMap().lookup(name);

            return genericClassInfo->classType;
        }

        if (getInterfacesMap().count(name))
        {
            auto interfaceInfo = getInterfacesMap().lookup(name);
            if (!interfaceInfo->interfaceType)
            {
                emitError(location) << "can't find interface: " << name << "\n";
                return mlir::Type();
            }

            // a @value interface is a value of its fields, not a reference seen through a vtable
            if (interfaceInfo->isValue)
            {
                return getValueTypeOfInterfaceIfLaidOut(interfaceInfo);
            }

            // `interface Cb { (x: number): number }` (and `extends Function`, an interface of nothing): one call
            // signature and nothing else is the function type, as `{ (x: number): number }` is. As an interface, a
            // function given to it was "invalid cast from (p0: number) => number to Cb" (TSLANG 175).
            // (not for a part of an intersection, which adds members to the interface: keepCallableInterface)
            auto callType = keepCallableInterface ? mlir::Type() : callSignatureOnlyType(interfaceInfo);
            keepCallableInterface = false;
            if (callType)
            {
                return callType;
            }

            return interfaceInfo->interfaceType;
        }

        if (getGenericInterfacesMap().count(name))
        {
            auto genericInterfaceInfo = getGenericInterfacesMap().lookup(name);
            return genericInterfaceInfo->interfaceType;
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return getEnumType(
                mlir::FlatSymbolRefAttr::get(builder.getContext(), concatFullNamespaceName(name)),
                enumTypeInfo.first, 
                enumTypeInfo.second);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto classInfo = getClassInfoByFullName(fullName);
            if (classInfo)
            {
                if (classInfo->isValue)
                {
                    return getValueTypeOfClassIfLaidOut(classInfo);
                }

                return classInfo->classType;
            }

            auto interfaceInfo = getInterfaceInfoByFullName(fullName);
            if (interfaceInfo)
            {
                if (interfaceInfo->isValue)
                {
                    return getValueTypeOfInterfaceIfLaidOut(interfaceInfo);
                }

                return interfaceInfo->interfaceType;
            }
        }

        // `import E = a.SomeEnum` as a type: the enum's (the referenced value's) type
        if (auto it = currentNamespace->importEqualsValueMap.find(name); it != currentNamespace->importEqualsValueMap.end())
        {
            // as evaluate() does, in a scratch module: a qualified name is not an expression mlirGen takes
            mlir::Type type;
            {
                TempModuleScope tempModuleScope(*this);
                SymbolTableScopeT varScope(symbolTable);
                GenContext evalGenContext(genContext);
                evalGenContext.allowPartialResolve = true;
                evalGenContext.funcOp = tempFuncOp;
                auto result = mlirGenModuleReference(it->second, evalGenContext);
                if (!result.failed_or_no_value())
                {
                    type = V(result).getType();
                }
            }

            if (type)
            {
                return type;
            }
        }

        // `import { T as U }`: U is T - one step, so `import { a as b, b as a }` cannot loop
        if (auto target = resolveImportAlias(name); !target.empty() && !resolvingImportAlias)
        {
            MLIRValueGuard<bool> aliasGuard(resolvingImportAlias, true);
            return resolveTypeByNameInNamespace(location, target, genContext);
        }

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::resolveTypeByName(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto type = resolveTypeByNameInNamespace(location, name, genContext);
        if (type)
        {
            return type;
        }

        {
            MLIRNamespaceGuard ng(currentNamespace);

            // search in outer namespaces
            while (currentNamespace->isFunctionNamespace)
            {
                currentNamespace = currentNamespace->parentNamespace;
                type = resolveTypeByNameInNamespace(location, name, genContext);
                if (type)
                {
                    return type;
                }
            }

            // search in root namespace
            currentNamespace = rootNamespace;
            type = resolveTypeByNameInNamespace(location, name, genContext);
            if (type)
            {
                return type;
            }
        }    

        if (!isEmbededType(name) && !circularTypeAliases.contains(name) && !valueTypesHoldingThemselves.contains(name))
            emitError(location, "can't find type by name: ") << name;

        return mlir::Type();    
    }

    mlir::Value MLIRGenImpl::resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (getGenericFunctionMap().count(name))
        {
            auto genericFunctionInfo = getGenericFunctionMap().lookup(name);

            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, genericFunctionInfo->funcType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericFunctionInfo->name));
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return funcSymbolOp;
        }

        auto value = resolveFunctionNameInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        if (getGlobalsMap().count(name))
        {
            auto value = getGlobalsMap().lookup(name);
            return globalVariableAccess(location, value, false, genContext);
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return builder.create<mlir_ts::ConstantOp>(
                location, 
                getEnumType(
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), concatFullNamespaceName(name)),
                    enumTypeInfo.first, 
                    enumTypeInfo.second), 
                enumTypeInfo.second);
        }

        // a namespace merged with a class of its name (`class C {} namespace C { export var x }`): the name is the
        // class, whose members a namespace member is found after (`C.x`, mlirGenPropertyAccessExpression). As the
        // namespace, `new C()` was "'C' is a namespace, not a class".
        if (getNamespaceMap().count(name) && !getClassesMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            assert(namespaceInfo);
            auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
            return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            // a class merged with a namespace of its name (`class C {} namespace C {}`, `import A = M.C`): the name is
            // the class, whose members a namespace member is found after (`new A()` was "'M.C' is a namespace, not a
            // class", TSLANG 191)
            if (auto classInfo = getClassInfoByFullName(fullName); classInfo && classInfo->classType)
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));
            }

            auto namespaceInfo = getNamespaceByFullName(fullName);
            if (namespaceInfo)
            {
                assert(namespaceInfo);
                auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
                return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
            }
        }

        if (auto it = currentNamespace->importEqualsValueMap.find(name); it != currentNamespace->importEqualsValueMap.end())
        {
            auto result = mlirGenModuleReference(it->second, genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Value();
            }

            return V(result);
        }

        // `import { a as b }`: b is a - one step, so `import { a as b, b as a }` cannot loop.
        // Before the type lookup below, which follows the alias too and would make an aliased
        // enum or class the type rather than the value.
        if (auto target = resolveImportAlias(name); !target.empty() && !resolvingImportAlias)
        {
            MLIRValueGuard<bool> aliasGuard(resolvingImportAlias, true);
            return resolveIdentifierInNamespace(location, target, genContext);
        }

        auto type = resolveTypeByNameInNamespace(location, name, genContext);

        // a type parameter is no value: `new A()` in `<A,>(x: A) => new A()` is the class `A` of this namespace or one
        // around it, not the type the parameter is given ("'new' on a value of type 'any'", TSLANG 191). One given a
        // class is that class, as before.
        if (type && !isa<mlir_ts::ClassType>(type) && genContext.typeParamsWithArgs.count(name))
        {
            MLIRNamespaceGuard ng(currentNamespace);
            for (auto namespaceInfo = currentNamespace; namespaceInfo; namespaceInfo = namespaceInfo->parentNamespace)
            {
                currentNamespace = namespaceInfo;
                if (auto classInfo = getClassesMap().count(name) ? getClassesMap().lookup(name) : ClassInfo::TypePtr();
                    classInfo && classInfo->classType && !classInfo->isValue)
                {
                    type = classInfo->classType;
                    break;
                }
            }
        }

        // as a value, a @value class's name is the class (`V.plus`, `new V()`), as any class's is -
        // before its fields are laid out too, when it has no type yet
        if (auto valueClassInfo = getValueClassInfo(type))
        {
            type = valueClassInfo->classType;
        }
        else if (auto classInfo = !type && getClassesMap().count(name) ? getClassesMap().lookup(name) : ClassInfo::TypePtr();
                 classInfo && classInfo->isValue && classInfo->classType)
        {
            type = classInfo->classType;
        }

        if (type)
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classType, mlir::FlatSymbolRefAttr::get(builder.getContext(), classType.getName().getValue()));
            }

            if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
            {
                return builder.create<mlir_ts::InterfaceRefOp>(
                    location, interfaceType, mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceType.getName().getValue()));
            }

            return builder.create<mlir_ts::TypeRefOp>(location, type);
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess,
                                          const GenContext &genContext)
    {
        if (fullNameGlobalsMap.count(name))
        {
            auto value = fullNameGlobalsMap.lookup(name);
            return globalVariableAccess(location, value, asAddess, genContext);
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifierAsVariable(location, name, genContext);
        if (value)
        {
            return value;
        }

        value = resolveIdentifierInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        {
            MLIRNamespaceGuard ng(currentNamespace);

            // search in outer namespaces
            while (currentNamespace->isFunctionNamespace)
            {
                currentNamespace = currentNamespace->parentNamespace;
                value = resolveIdentifierInNamespace(location, name, genContext);
                if (value)
                {
                    return value;
                }
            }

            // search in root namespace
            currentNamespace = rootNamespace;
            value = resolveIdentifierInNamespace(location, name, genContext);
            if (value)
            {
                return value;
            }
        }

        // try to resolve 'this' if not resolved yet
        if (genContext.thisType && name == THIS_NAME)
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(genContext.thisType)) {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classType, mlir::FlatSymbolRefAttr::get(builder.getContext(), 
                    classType.getName().getValue()));
            }

            return builder.create<mlir_ts::TypeRefOp>(location, genContext.thisType);
        }

        if (genContext.thisType && name == SUPER_NAME)
        {
            mlir::Value thisValue;
            auto thisType = genContext.thisType;
            if (!isa<mlir_ts::ClassType>(genContext.thisType) && !isa<mlir_ts::ClassStorageType>(genContext.thisType))
            {
                auto result = mlirGen(location, THIS_ALIAS, genContext);
                if (result.failed_or_no_value()) {
                    return mlir::Value();
                }

                thisValue = V(result);
                thisType = thisValue.getType();
                if (!isa<mlir_ts::ClassType>(thisType) && !isa<mlir_ts::ClassStorageType>(thisType)) {
                    return mlir::Value();
                }
            }
            else
            {
                auto result = mlirGen(location, THIS_NAME, genContext);
                thisValue = V(result);
            }

            auto fullName = isa<mlir_ts::ClassStorageType>(thisType) 
                ? mlir::cast<mlir_ts::ClassStorageType>(thisType).getName().getValue() 
                : mlir::cast<mlir_ts::ClassType>(thisType).getName().getValue();
            auto classInfo = getClassInfoByFullName(fullName);
            // `super` in a class that extends nothing (`superNoModifiersCrash.ts`): read the base of an empty list
            if (!classInfo || classInfo->baseClasses.empty())
            {
                emitError(location) << "'super' can only be used in a class that extends another";
                return mlir::Value();
            }

            auto baseClassInfo = classInfo->baseClasses.front();

            // this is access to static base class
            if (thisValue.getDefiningOp<mlir_ts::ClassRefOp>())
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, baseClassInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                baseClassInfo->classType.getName().getValue()));                   
            }

            return mlirGenPropertyAccessExpression(location, thisValue, baseClassInfo->fullName, genContext);
        }

        // `super` in an arrow in a method (`() => super.m()`): the arrow has the method's `this`, captured, and
        // no type of `this` of its own; `super` is that object's base (it was "can't resolve name: super")
        if (!genContext.thisType && name == SUPER_NAME)
        {
            if (auto thisValue = resolveIdentifier(location, THIS_NAME, genContext))
            {
                auto thisType = thisValue.getType();
                if (auto refType = dyn_cast<mlir_ts::RefType>(thisType))
                {
                    thisType = refType.getElementType();
                    thisValue = builder.create<mlir_ts::LoadOp>(location, thisType, thisValue);
                }

                if (auto classType = dyn_cast<mlir_ts::ClassType>(thisType))
                {
                    auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                    if (classInfo && !classInfo->baseClasses.empty())
                    {
                        return mlirGenPropertyAccessExpression(location, thisValue, classInfo->baseClasses.front()->fullName,
                                                               genContext);
                    }
                }
            }
        }

        // built-in types
        if (name == UNDEFINED_NAME)
        {
            return getUndefined(location);
        }

        if (name == INFINITY_NAME)
        {
            return getInfinity(location);
        }

        if (name == NAN_NAME)
        {
            return getNaN(location);
        }

        // end of built-in types

        value = resolveFullNameIdentifier(location, name, false, genContext);
        if (value)
        {
            return value;
        }

        return mlir::Value();
    }

} // namespace mlirgen
} // namespace typescript
