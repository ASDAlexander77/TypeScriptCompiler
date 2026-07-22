// Module, discovery, include/import driver methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "TypeScript/ObjDumper.h"

#include "MLIRGenImpl.h"



#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ToolOutputFile.h"

#include <set>

namespace typescript
{
namespace mlirgen
{

    mlir::LogicalResult MLIRGenImpl::report(SourceFile module, const std::vector<SourceFile> &includeFiles)
    {
        // output diag info
        auto hasAnyError = false;
        auto fileName = convertWideToUTF8(module->fileName);
        for (auto diag : module->parseDiagnostics)
        {
            hasAnyError |= diag.category == DiagnosticCategory::Error;
            if (diag.category == DiagnosticCategory::Error)
            {
                emitError(loc2(module, fileName, diag.start, diag.length), convertWideToUTF8(diag.messageText));
            }
            else
            {
                emitWarning(loc2(module, fileName, diag.start, diag.length), convertWideToUTF8(diag.messageText));
            }
        }

        for (auto incFile : includeFiles)
        {
            auto fileName = convertWideToUTF8(incFile->fileName);
            for (auto diag : incFile->parseDiagnostics)
            {
                hasAnyError |= diag.category == DiagnosticCategory::Error;
                if (diag.category == DiagnosticCategory::Error)
                {
                    emitError(loc2(incFile, fileName, diag.start, diag.length), convertWideToUTF8(diag.messageText));
                }
                else
                {
                    emitWarning(loc2(incFile, fileName, diag.start, diag.length), convertWideToUTF8(diag.messageText));
                }
            }
        }

        return hasAnyError ? mlir::failure() : mlir::success();
    }

    std::pair<SourceFile, std::vector<SourceFile>> MLIRGenImpl::loadMainSourceFile()
    {
        const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        auto sourceFileLoc = mlir::FileLineColLoc::get(builder.getContext(),
                    sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
        return loadSourceBuf(sourceFileLoc, sourceBuf, true);
    }    

    std::pair<SourceFile, std::vector<SourceFile>> MLIRGenImpl::loadSourceFile(SMLoc loc)
    {
        const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.FindBufferContainingLoc(loc));
        auto sourceFileLoc = mlir::FileLineColLoc::get(builder.getContext(),
                    sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
        return loadSourceBuf(sourceFileLoc, sourceBuf, true);
    }        

    std::pair<SourceFile, std::vector<SourceFile>> MLIRGenImpl::loadSourceBuf(mlir::Location location, const llvm::MemoryBuffer *sourceBuf, bool isMain)
    {
        std::vector<SourceFile> includeFiles;
        std::vector<string> filesToProcess;

        LocationHelper lh(builder.getContext());

        auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(location);
        auto dirName = file.getDirectory();
        auto sourceFileName = file.getName();

        SmallString<256> fullPath;
        sys::path::append(fullPath, dirName.getValue());
        sys::path::append(fullPath, sourceFileName.getValue());

        auto fullPathW = stows(fullPath.str().str());

        Parser parser;
        auto sourceFile = parser.parseSourceFile(
            fullPathW, 
            stows(sourceBuf->getBuffer().str()), 
            ScriptTarget::Latest);
        sourceFile->resolvedPath = fullPathW;

        // add default lib
        if (isMain)
        {
            if (sourceFile->hasNoDefaultLib)
            {
                compileOptions.noDefaultLib = true;
            }

            if (!compileOptions.noDefaultLib)
            {
                //  S(DEFAULT_LIB_DIR "/lib.d.ts")
                filesToProcess.push_back(convertUTF8toWide(compileOptions.defaultDeclarationTSFile));
            }

            auto strictNull = sourceFile->pragmas.find(S("strict-null"));
            if (strictNull != sourceFile->pragmas.end())
            {                
                auto option = strictNull->second.front().find(S("option"));
                if (option != strictNull->second.front().end()) 
                {
                    compileOptions.strictNullChecks = option->second._arg.value == S("true");
                }
            }
        }

        for (auto refFile : sourceFile->referencedFiles)
        {
            filesToProcess.push_back(refFile.fileName);
        }

        while (filesToProcess.size() > 0)
        {
            auto includeFileName = filesToProcess.back();
            auto includeFileNameUtf8 = convertWideToUTF8(includeFileName);
            filesToProcess.pop_back();

            std::string actualFilePath;
            auto id = sourceMgr.AddIncludeFile(std::string(includeFileNameUtf8), SMLoc(), actualFilePath);
            if (!id)
            {
                emitError(location, "can't open file: ") << fullPath;
                continue;
            }

            SmallString<256> fullPath;
            if (!sys::path::has_root_path(actualFilePath))
            {
                sys::path::append(fullPath, dirName.getValue());
            }

            sys::path::append(fullPath, actualFilePath);

            const auto *sourceBuf = sourceMgr.getMemoryBuffer(id);

            auto actualFilePathW = convertUTF8toWide(fullPath.str().str());

            Parser parser;
            auto includeFile =
                parser.parseSourceFile(
                    actualFilePathW, 
                    stows(sourceBuf->getBuffer().str()), 
                    ScriptTarget::Latest);
            includeFile->resolvedPath = actualFilePathW;

            for (auto refFile : includeFile->referencedFiles)
            {
                filesToProcess.push_back(refFile.fileName);
            }

            includeFiles.push_back(includeFile);
        }

        std::reverse(includeFiles.begin(), includeFiles.end());

        return {sourceFile, includeFiles};
    }

    mlir::LogicalResult MLIRGenImpl::showMessages(SourceFile module, std::vector<SourceFile> includeFiles)
    {
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
            sourceMgrHandler.emit(diag);
        });

        if (mlir::failed(report(module, includeFiles)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::ModuleOp MLIRGenImpl::mlirGenSourceFile(SourceFile module, std::vector<SourceFile> includeFiles)
    {
        if (mlir::failed(showMessages(module, includeFiles)))
        {
            return nullptr;
        }        

        DITableScopeT debugSourceFileScope(debugScope);
        if (mlir::failed(mlirGenCodeGenInit(module)))
        {
            return nullptr;
        }

        SymbolTableScopeT varScope(symbolTable);
        llvm::ScopedHashTableScope<StringRef, NamespaceInfo::TypePtr> fullNamespacesMapScope(fullNamespacesMap);
        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(
            fullNameGlobalsMap);
        llvm::ScopedHashTableScope<StringRef, GenericFunctionInfo::TypePtr> fullNameGenericFunctionsMapScope(
            fullNameGenericFunctionsMap);
        llvm::ScopedHashTableScope<StringRef, EnumInfo::TypePtr> fullNameEnumsMapScope(fullNameEnumsMap);
        llvm::ScopedHashTableScope<StringRef, ClassInfo::TypePtr> fullNameClassesMapScope(fullNameClassesMap);
        llvm::ScopedHashTableScope<StringRef, GenericClassInfo::TypePtr> fullNameGenericClassesMapScope(
            fullNameGenericClassesMap);
        llvm::ScopedHashTableScope<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMapScope(fullNameInterfacesMap);
        llvm::ScopedHashTableScope<StringRef, GenericInterfaceInfo::TypePtr> fullNameGenericInterfacesMapScope(
            fullNameGenericInterfacesMap);
        SafeTypesMapScopeT safeTypesMapScope(safeTypesMap);

        stage = Stages::Discovering;
        auto storeDebugInfo = compileOptions.generateDebugInfo;
        compileOptions.generateDebugInfo = false;
        if (mlir::succeeded(mlirDiscoverAllDependencies(module, includeFiles)))
        {
            stage = Stages::SourceGeneration;
            compileOptions.generateDebugInfo = storeDebugInfo;
            if (mlir::succeeded(mlirCodeGenModule(module, includeFiles)))
            {
                return theModule;
            }
        }

        return nullptr;
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenCodeGenInit(SourceFile module)
    {
        sourceFile = module;

        auto location = loc(module);
        if (compileOptions.generateDebugInfo)
        {
            auto isOptimized = false;

            MLIRDebugInfoHelper mdi(builder, debugScope);
            mdi.setFile(mainSourceFileName);
            location = mdi.getCompileUnit(location, "TypeScript Native Compiler", isOptimized);
        }

        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(location, mainSourceFileName);

        if (!compileOptions.moduleTargetTriple.empty())
        {
            theModule->setAttr(
                mlir::LLVM::LLVMDialect::getTargetTripleAttrName(), 
                builder.getStringAttr(compileOptions.moduleTargetTriple));

            // DataLayout for IndexType
            // TODO: seems u need to do it on LLVM level, as LLVMTypeHelper knows size of index
            auto indexSize = mlir::DataLayoutEntryAttr::get(builder.getIndexType(), builder.getI32IntegerAttr(compileOptions.sizeBits));
            theModule->setAttr("dlti.dl_spec", mlir::DataLayoutSpecAttr::get(builder.getContext(), {indexSize}));
        }

        builder.setInsertionPointToStart(theModule.getBody());

        return mlir::success();
    }

#ifdef GENERATE_IMPORT_INFO_USING_D_TS_FILE
    mlir::LogicalResult MLIRGenImpl::createDependencyDeclarationFile(StringRef outputFilename,
                                            StringRef dependencyDeclFileBody) {
        std::string errorMessage;
        std::unique_ptr<llvm::ToolOutputFile> outputFile =
            openOutputFile(outputFilename, &errorMessage);
        if (!outputFile) {
            llvm::errs() << errorMessage << "\n";
            return failure();
        }

        outputFile->os() << dependencyDeclFileBody << "\n";
        outputFile->keep();

        return success();
    }

#endif    
    mlir::LogicalResult MLIRGenImpl::createDeclarationExportGlobalVar(const GenContext &genContext)
    {
        if (!declExports.rdbuf()->in_avail() || !compileOptions.embedExportDeclarations)
        {
            return mlir::success();
        }

#ifdef GENERATE_IMPORT_INFO_USING_D_TS_FILE
        if (mainSourceFileName == SHARED_LIB_DECLARATIONS_FILENAME)
        {
            return mlir::success();
        }
#endif        

        auto declText = declExports.str();

#ifndef GENERATE_IMPORT_INFO_USING_D_TS_FILE
        // default implementation to use variable to store declaration data
        LLVM_DEBUG(llvm::dbgs() << "\n!! export declaration: \n" << declText << "\n";);

        auto typeWithInit = [&](mlir::Location location, const GenContext &genContext) {
            auto litValue = V(mlirGenStringValue(location, declText, true));
            return std::make_tuple(litValue.getType(), litValue, TypeProvided::No);            
        };

        auto loc = mlir::UnknownLoc::get(builder.getContext());

        VariableClass varClass = VariableType::Var;
        varClass.isExport = true;
        varClass.isPublic = true;

        std::string varName(SHARED_LIB_DECLARATIONS_2UNDERSCORE);
        varName.append("_");
        varName.append(llvm::sys::path::stem(llvm::sys::path::filename(mainSourceFileName)));
        varName.append("_");
        varName.append(to_string(hash_value(mainSourceFileName)));
        
        auto varNameRef = StringRef(varName).copy(stringAllocator);
        
        auto varType = registerVariable(loc, varNameRef, true, varClass, typeWithInit, genContext);
#endif        

#ifdef GENERATE_IMPORT_INFO_USING_D_TS_FILE
        llvm::SmallString<128> path(compileOptions.outputFolder);
        llvm::sys::path::append(path, llvm::sys::path::filename(mainSourceFileName));
        llvm::sys::path::replace_extension(path, ".d.ts");
        return createDependencyDeclarationFile(path, declText);
#else
        return success();
#endif
    }

    mlir::LogicalResult MLIRGenImpl::createGenericClassDeclarationExportGlobalVar(const GenContext &genContext)
    {
        if (!genericDeclExports.rdbuf()->in_avail() || !compileOptions.embedExportDeclarations)
        {
            return mlir::success();
        }

        auto declText = genericDeclExports.str();

        LLVM_DEBUG(llvm::dbgs() << "\n!! export generic class declaration: \n" << declText << "\n";);

        auto typeWithInit = [&](mlir::Location location, const GenContext &genContext) {
            auto litValue = V(mlirGenStringValue(location, declText, true));
            return std::make_tuple(litValue.getType(), litValue, TypeProvided::No);
        };

        auto loc = mlir::UnknownLoc::get(builder.getContext());

        VariableClass varClass = VariableType::Var;
        varClass.isExport = true;
        varClass.isPublic = true;

        // "generic" in the middle keeps the "__decls" prefix (so the existing
        // symbol.starts_with(SHARED_LIB_DECLARATIONS_2UNDERSCORE) enumeration in
        // mlirGenImportSharedLib still finds it) while staying distinguishable from the
        // regular per-file "__decls_<file>_<hash>" global, so that call site can tell the
        // two apart and parse each with the right file_d_ts flag.
        std::string varName(SHARED_LIB_DECLARATIONS_2UNDERSCORE);
        varName.append("_generic_");
        varName.append(llvm::sys::path::stem(llvm::sys::path::filename(mainSourceFileName)));
        varName.append("_");
        varName.append(to_string(hash_value(mainSourceFileName)));

        auto varNameRef = StringRef(varName).copy(stringAllocator);

        registerVariable(loc, varNameRef, true, varClass, typeWithInit, genContext);

        return mlir::success();
    }

    bool MLIRGenImpl::isCodeStatment(SyntaxKind kind)
    {
        static std::set<SyntaxKind> codeStatements {
            SyntaxKind::ExpressionStatement,
            SyntaxKind::IfStatement,
            SyntaxKind::ReturnStatement,
            SyntaxKind::LabeledStatement,
            SyntaxKind::DoStatement,
            SyntaxKind::WhileStatement,
            SyntaxKind::ForStatement,
            SyntaxKind::ForInStatement,
            SyntaxKind::ForOfStatement,
            SyntaxKind::ContinueStatement,
            SyntaxKind::BreakStatement,
            SyntaxKind::SwitchStatement,
            SyntaxKind::ThrowStatement,
            SyntaxKind::TryStatement,
            SyntaxKind::Block,
            SyntaxKind::DebuggerStatement
        };

        return codeStatements.find(kind) != codeStatements.end();    
    }

    int MLIRGenImpl::processStatements(NodeArray<Statement> statements,
                          const GenContext &genContext,
                          bool isRoot)
    {
        clearState(statements);

        auto notResolved = 0;
        do
        {
            // main cycles
            auto noErrorLocation = true;
            mlir::Location errorLocation = mlir::UnknownLoc::get(builder.getContext());
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            // clear previous errors
            genContext.postponedMessages->clear();
            for (auto &statement : statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (isRoot && (isCodeStatment(statement) || statement == SyntaxKind::VariableStatement))
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    emitError(loc(statement), "failed statement");

                    notResolved++;
                    if (noErrorLocation)
                    {
                        errorLocation = loc(statement);
                        noErrorLocation = false;
                    }

                    if (genContext.isStopped())
                    {
                        return notResolved;
                    }
                }
                else
                {
                    statement->processed = true;
                }
            }

            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                break;
            }

        } while (notResolved > 0);

        // clear states to be able to run second time
        clearState(statements);
        
        return notResolved;
    }

    bool MLIRGenImpl::hasGlobalCode(NodeArray<Statement> statements) {
        auto anyCode = false;
        for (auto &statement : statements)
        {
            if (isCodeStatment(statement))
            {
                anyCode = true;
                break;
            }
        }

        return anyCode;        
    }

    void MLIRGenImpl::addGlobalConstructor(mlir::Location location, StringRef funcName)
    {
        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        MLIRCodeLogicHelper mclh(builder, location, compileOptions);

        builder.setInsertionPointToStart(theModule.getBody());
        mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(theModule.getBody());

        builder.create<mlir_ts::GlobalConstructorOp>(
            location, mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName),
            builder.getIndexAttr(LAST_GLOBAL_CONSTRUCTOR_PRIORITY));
    }

    mlir::LogicalResult MLIRGenImpl::generateGlobalEntryCode(mlir::Location location, NodeArray<Statement> statements,
                          const GenContext &genContext)
    {
        // create function
        //auto name = MLIRHelper::getAnonymousName(location, ".main", "");
        auto useGlobalCtor = false;
        std::string name = MAIN_ENTRY_NAME;
        auto fullGlobalFuncName = getFullNamespaceName(name);

        if (theModule.lookupSymbol(fullGlobalFuncName))
        {
            // create global ctor
            name = MLIRHelper::getAnonymousName(location, "." MAIN_ENTRY_NAME, "");
            fullGlobalFuncName = getFullNamespaceName(name);
            useGlobalCtor = true;
        }

        mlir::OpBuilder::InsertionGuard insertGuard(builder);

        // create global construct
        auto funcType = getFunctionType({}, {}, false);

        if (mlir::failed(mlirGenFunctionBody(location, name, fullGlobalFuncName, funcType,
            [&](mlir::Location location, const GenContext &genContext) {
                for (auto &statement : statements)
                {
                    auto isVariableStatement = statement == SyntaxKind::VariableStatement;
                    if (isCodeStatment(statement) || isVariableStatement)
                    {
                        if (isVariableStatement)
                        {
                            // patch VariableStatement
                            auto variableStatement = statement.as<VariableStatement>();
                            variableStatement->declarationList->flags &= ~NodeFlags::Let;
                            auto hasArrowDeclaration = llvm::any_of(
                                variableStatement->declarationList->declarations, 
                                [](auto decl) { return decl->initializer == SyntaxKind::ArrowFunction; });
                            if (!hasArrowDeclaration)
                            {
                                variableStatement->declarationList->flags &= ~NodeFlags::Const;                        
                            }
                        }

                        if (failed(mlirGen(statement, genContext)))
                        {
                            emitError(loc(statement), "failed statement");
                            return mlir::failure();
                        }
                    }

                }

                return mlir::success();
            }, genContext, 0, true)))
        {
            return mlir::failure();
        }

        if (useGlobalCtor)
        {
            addGlobalConstructor(location, fullGlobalFuncName);
        }
        
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::outputDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages,
                                          int notResolved)
    {
        // print errors
        if (notResolved)
        {
            printDiagnostics(sourceMgrHandler, postponedMessages, compileOptions.disableWarnings);
        }

        postponedMessages.clear();

        // we return error when we can't generate code
        if (notResolved)
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirDiscoverAllDependencies(SourceFile module, std::vector<SourceFile> includeFiles)
    {
        mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedMessages;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
            postponedMessages.emplace_back(new mlir::Diagnostic(std::move(diag)));
        });

        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(
            fullNameGlobalsMap);

        // Discovery emits into a throwaway module, so its cleanup can never disturb real module
        // content. When this discovery pass is nested (an 'import' of a local source file triggers
        // mlirGenInclude during SourceGeneration), the real module already holds generated content
        // (e.g. default-lib function bodies such as 'console.log') that must survive.
        DiscoveryModuleScope discoveryModuleScope(*this);

        // Process of discovery here
        GenContext genContextPartial{};
        genContextPartial.allowPartialResolve = true;
        genContextPartial.dummyRun = true;
        genContextPartial.rootContext = &genContextPartial;
        genContextPartial.postponedMessages = &postponedMessages;

        for (auto includeFile : includeFiles)
        {
            SourceFileScope sourceFileScope(*this, includeFile);

            if (failed(mlirGen(includeFile->statements, genContextPartial)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }
        }

        auto notResolved = processStatements(module->statements, genContextPartial);

        // clean up; the ops this pass created go away with the discovery module on scope exit
        clearTempModule();

        // clear state
        for (auto &statement : module->statements)
        {
            statement->processed = false;
        }

        if (failed(outputDiagnostics(postponedMessages, notResolved)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirCodeGenModule(SourceFile module, std::vector<SourceFile> includeFiles,
                                          bool validate, bool isMain)
    {
        mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedWarningsMessages;
        mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedMessages;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            {
                postponedMessages.emplace_back(new mlir::Diagnostic(std::move(diag)));
            }
            else
            {
                postponedWarningsMessages.emplace_back(new mlir::Diagnostic(std::move(diag)));
            }
        });

        // Process generating here
        declExports.str("");
        declExports.clear();
        GenContext genContext{};
        genContext.rootContext = &genContext;
        genContext.postponedMessages = &postponedMessages;

        for (auto includeFile : includeFiles)
        {
            SourceFileScope sourceFileScope(*this, includeFile);

            if (failed(mlirGen(includeFile->statements, genContext)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }
        }

        auto anyGlobalCode = hasGlobalCode(module->statements);
        auto notResolved = processStatements(module->statements, genContext, isMain && anyGlobalCode);       
        if (failed(outputDiagnostics(postponedMessages, notResolved)))
        {
            return mlir::failure();
        }
       
        if (isMain && notResolved == 0)
        {
            // generate code to run at global entry
            if (anyGlobalCode && mlir::failed(
                generateGlobalEntryCode(loc(module), module->statements, genContext)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }

            // exports
            if (mlir::failed(createDeclarationExportGlobalVar(genContext))) {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }

            if (mlir::failed(createGenericClassDeclarationExportGlobalVar(genContext))) {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }
        }

        clearTempModule();

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the TypeScript operations.
        if (validate && failed(mlir::verify(theModule)))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! broken module: \n" << theModule << "\n";);

            theModule.emitError("module verification error");

            // to show all errors now
            outputDiagnostics(postponedMessages, 1);
            return mlir::failure();
        }

        printDiagnostics(sourceMgrHandler, postponedWarningsMessages, compileOptions.disableWarnings);

        return mlir::success();
    }

    bool MLIRGenImpl::registerNamespace(llvm::StringRef namePtr, bool isFunctionNamespace)
    {
        if (isFunctionNamespace)
        {
            std::string res;
            res += ".f_";
            res += namePtr;
            namePtr = StringRef(res).copy(stringAllocator);
        }
        else
        {
            namePtr = StringRef(namePtr).copy(stringAllocator);
        }

        auto fullNamePtr = getFullNamespaceName(namePtr);
        auto &namespacesMap = getNamespaceMap();
        auto it = namespacesMap.find(namePtr);
        if (it == namespacesMap.end())
        {
            auto newNamespacePtr = std::make_shared<NamespaceInfo>();
            newNamespacePtr->name = namePtr;
            newNamespacePtr->fullName = fullNamePtr;
            newNamespacePtr->namespaceType = getNamespaceType(fullNamePtr);
            newNamespacePtr->parentNamespace = currentNamespace;
            newNamespacePtr->isFunctionNamespace = isFunctionNamespace;

            namespacesMap.insert({namePtr, newNamespacePtr});
            if (!isFunctionNamespace && !fullNamespacesMap.count(fullNamePtr))
            {
                // TODO: full investigation needed, if i register function namespace as full namespace, it will fail
                // running
                fullNamespacesMap.insert(fullNamePtr, newNamespacePtr);
            }

            currentNamespace = newNamespacePtr;
        }
        else
        {
            currentNamespace = it->getValue();
            return false;
        }

        return true;
    }

    mlir::LogicalResult MLIRGenImpl::exitNamespace()
    {
        // TODO: it will increase reference count, investigate how to fix it
        currentNamespace = currentNamespace->parentNamespace;
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(moduleDeclarationAST);

        auto namespaceName = MLIRHelper::getName(moduleDeclarationAST->name, stringAllocator);
        auto namePtr = namespaceName;

        MLIRNamespaceGuard nsGuard(currentNamespace);
        registerNamespace(namePtr);

        DITableScopeT debugNamespaceScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            mdi.setNamespace(location, namePtr, hasModifier(moduleDeclarationAST, SyntaxKind::ExportKeyword));
        }

        return mlirGenBody(moduleDeclarationAST->body, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        return mlirGenNamespace(moduleDeclarationAST, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenInclude(mlir::Location location, StringRef filePath, const GenContext &genContext)
    {
        MLIRValueGuard<bool> vg(declarationMode);
        declarationMode = true;

        auto [importSource, importIncludeFiles] = loadIncludeFile(location, filePath);
        if (!importSource)
        {
            return mlir::failure();
        }

        if (mlir::failed(showMessages(importSource, importIncludeFiles)))
        {
            return mlir::failure();
        }          

        // we need to override filename to track it in DBG info
        SourceFileScope sourceFileScope(*this, importSource);

        if (mlir::succeeded(mlirDiscoverAllDependencies(importSource, importIncludeFiles)) &&
            mlir::succeeded(mlirCodeGenModule(importSource, importIncludeFiles, false, false)))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenImportSharedLib(mlir::Location location, StringRef filePath, bool dynamic, const GenContext &genContext)
    {
        // TODO: ...
        std::string errMsg;
        auto dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(filePath.str().c_str(), &errMsg);
        if (!dynLib.isValid())
        {
            emitError(location, errMsg);
            return mlir::failure();
        }

        SmallVector<StringRef> symbols;
        StringRef mlirGctors;
#ifndef GENERATE_IMPORT_INFO_USING_D_TS_FILE
        // loading Binary to get list of symbols
        SmallVector<StringRef> symbolsAll;
        Dump::getSymbols(filePath, symbolsAll, stringAllocator);

        for (auto symbol : symbolsAll)
        {
            if (symbol.starts_with(SHARED_LIB_DECLARATIONS_2UNDERSCORE))
            {
                symbols.push_back(symbol);
            }
            else if (symbol == MLIR_GCTORS)
            {
                mlirGctors = symbol;
            }
        }
#else
        // only 1 file to load        
        symbols.push_back(SHARED_LIB_DECLARATIONS_2UNDERSCORE);
#endif        

        if (symbols.empty())
        {
            emitWarning(location, "missing information about shared library. (reference " SHARED_LIB_DECLARATIONS " is missing)");            
        }

        // load library
        auto name = MLIRHelper::getAnonymousName(location, ".ll", "");
        auto fullInitGlobalFuncName = getFullNamespaceName(name);

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            // create global construct
            auto funcType = getFunctionType({}, {}, false);

            if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, funcType,
                [&](mlir::Location location, const GenContext &genContext) {
                    auto litValue = mlirGenStringValue(location, filePath.str());
                    auto strVal = cast(location, getStringType(), litValue, genContext);
                    builder.create<mlir_ts::LoadLibraryPermanentlyOp>(location, mth.getI32Type(), strVal);

                    // call global inits
                    if (!mlirGctors.empty())
                    {
                        auto mlirGctorsNameVal = mlirGenStringValue(location, mlirGctors);
                        auto strVal = cast(location, getStringType(), mlirGctorsNameVal, genContext);                        
                        auto globalCtorPtr = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                            location, mlir_ts::OpaqueType::get(builder.getContext()), strVal);
                        auto funcPtr = builder.create<mlir_ts::CastOp>(location, getFunctionType({}, {}, false), globalCtorPtr);
                        builder.create<mlir_ts::CallIndirectOp>(location, funcPtr, mlir::ValueRange{});                        
                    }

                    return mlir::success();
                }, genContext)))
            {
                return mlir::failure();
            }

            // The shared-lib load + symbol resolution call into LLVM's
            // sys::DynamicLibrary, which uses std::vector. In debug builds STL
            // iterators take a global lock that the CRT only initializes via its
            // own '_Init_locks'/'initlocks' dynamic initializer (in .CRT$XCU).
            // FIRST_GLOBAL_CONSTRUCTOR_PRIORITY (100) places this ctor BEFORE that
            // CRT init -> entering an uninitialized CRITICAL_SECTION -> crash.
            // Use the same band as the per-symbol __cctors (LAST) so it runs after
            // 'initlocks'; it is emitted before them, so it still loads the library
            // before any LLVMSearchForAddressOfSymbol runs.
            addGlobalConstructor(location, fullInitGlobalFuncName);
        }

        for (auto declSymbol : symbols)
        {
            // TODO: for now, we have code in TS to load methods from DLL/Shared libs
            if (auto addrOfDeclText = dynLib.getAddressOfSymbol(declSymbol.str().c_str()))
            {
                std::string result;
                // process shared lib declarations
                auto dataPtr = *(const char**)addrOfDeclText;
                if (dynamic)
                {
                    // TODO: use option variable instead of "this hack"
                    result = MLIRHelper::replaceAll(dataPtr, "@dllimport", "@dllimport('.')");
                    dataPtr = result.c_str();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! Shared lib import: \n" << dataPtr << "\n";);

                {
                    MLIRLocationGuard vgLoc(overwriteLoc);
                    overwriteLoc = location;

                    // a generic class's declaration (see
                    // createGenericClassDeclarationExportGlobalVar) must NOT be parsed with
                    // the ".d.ts" filename convention used below for every other kind of
                    // declaration: that convention makes the parser mark everything ambient/
                    // external regardless of any per-declaration @dllimport marker (see the
                    // comment on parsePartialStatements's file_d_ts parameter), which would
                    // make the generic's instantiated specializations wrongly look like
                    // external stubs with no compilable body - they need to be treated as
                    // ordinary, fully-compilable local source instead.
                    auto isGenericClassDecl =
                        declSymbol.starts_with(std::string(SHARED_LIB_DECLARATIONS_2UNDERSCORE) + "_generic_");

                    auto importData = convertUTF8toWide(dataPtr);
                    if (mlir::failed(parsePartialStatements(importData, genContext, false, !isGenericClassDecl)))
                    {
                        //assert(false);
                        return mlir::failure();
                    }
                }
            }
            else
            {
                emitWarning(location, "missing information about shared library. (reference " SHARED_LIB_DECLARATIONS " is missing)");
            }
        }

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGen(ImportDeclaration importDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(importDeclarationAST);

        auto result = mlirGen(importDeclarationAST->moduleSpecifier, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto modulePath = V(result);

        auto constantOp = modulePath.getDefiningOp<mlir_ts::ConstantOp>();
        assert(constantOp);
        auto valueAttr = mlir::cast<mlir::StringAttr>(constantOp.getValueAttr());

        auto stringVal = valueAttr.getValue();

        std::string fullPath;
        fullPath += stringVal;
#ifdef WIN_LOADSHAREDLIBS
#endif        
#ifdef LINUX_LOADSHAREDLIBS
        // rebuild file path
        auto fileName = sys::path::filename(stringVal);
        auto path = stringVal.substr(0, stringVal.size() - fileName.size());
        fullPath = path;
        fullPath += "lib";
        fullPath += fileName;
#endif

        if (sys::path::extension(fullPath) == "")
        {
#ifdef WIN_LOADSHAREDLIBS
            fullPath += ".dll";
#endif
#ifdef LINUX_LOADSHAREDLIBS
            fullPath += ".so";
#endif
        }

        if (sys::fs::exists(fullPath))
        {
            //auto dynamic = MLIRHelper::hasDecorator(importDeclarationAST, "dynamic");
            auto dynamic = !MLIRHelper::hasDecorator(importDeclarationAST, "static");

            // this is shared lib.
            return mlirGenImportSharedLib(location, fullPath, dynamic, genContext);    
        }

        return mlirGenInclude(location, stringVal, genContext);
    }

} // namespace mlirgen
} // namespace typescript
