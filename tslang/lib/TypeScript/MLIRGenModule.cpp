// Module, discovery, include/import driver methods of MLIRGenImpl (see MLIRGenImpl.h).

// Included first, before MLIRGenImpl.h: MLIRGenImpl.h transitively pulls in ts-new-parser's
// config.h and MLIRGenContextDefines.h, which #define single-letter macros (S(x), V(x), ...)
// for terse AST/MLIR construction. llvm/Support/CommandLine.h and FormattedStream.h,
// transitively included by the two headers below, use those same letters as ordinary
// identifiers (parameter and member names), so those macros corrupt them if already active.
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include "TypeScript/ObjDumper.h"

#include "MLIRGenImpl.h"



#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/xxhash.h"
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

    std::string MLIRGenImpl::canonicalFilePath(StringRef filePath)
    {
        // real_path also folds a Windows spelling that differs only in case
        SmallString<256> canonical;
        if (!sys::fs::real_path(filePath, canonical))
        {
            return canonical.str().str();
        }

        // no such file: the path made absolute, with `.` and `..` removed
        canonical = filePath;
        sys::fs::make_absolute(canonical);
        sys::path::remove_dots(canonical, /*remove_dot_dot=*/true);
        return canonical.str().str();
    }

    std::string MLIRGenImpl::moduleSymbolSuffix(StringRef filePath)
    {
        std::string suffix(llvm::sys::path::stem(llvm::sys::path::filename(filePath)));
        suffix.append("_");
        suffix.append(to_string(llvm::xxh3_64bits(canonicalFilePath(filePath))));
        return suffix;
    }

    static std::string fileKey(SourceFile sourceFile)
    {
        return convertWideToUTF8(sourceFile->resolvedPath);
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
        auto sourceFilePath = canonicalFilePath(fullPath);
        sourceFile->resolvedPath = convertUTF8toWide(sourceFilePath);

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

        // Every file referenced, directly or through another, is loaded once, and comes after the
        // files it references, so a declaration precedes its uses: for a diamond (b and c both
        // reference common) that is common, b, c. The file being loaded is already seen, so a
        // reference back to it, or a reference cycle, ends there.
        llvm::StringSet<> seenFiles;
        seenFiles.insert(sourceFilePath);

        // A file that is not there fails the load: the program would otherwise be compiled
        // without the declarations it asked for.
        auto anyMissing = false;

        // `referencing` holds the reference (null for the default lib), in the directory
        // `referencingDir`: a relative path is that file's, as in TypeScript, so it is looked for
        // there first. Then as before - the working directory, the main file's directory, the
        // default lib's - which a reference relative to the main file from a file elsewhere
        // still relies on.
        std::function<void(const string &, SourceFile, const ts::data::FileReference *, StringRef)> loadReferencedFile =
            [&](const string &includeFileName, SourceFile referencing, const ts::data::FileReference *reference, StringRef referencingDir) {
            auto includeFileNameUtf8 = convertWideToUTF8(includeFileName);

            std::string actualFilePath;
            llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> includeBuf = std::make_error_code(std::errc::no_such_file_or_directory);
            if (!sys::path::has_root_path(includeFileNameUtf8) && !referencingDir.empty())
            {
                SmallString<256> besideReferencing(referencingDir);
                sys::path::append(besideReferencing, includeFileNameUtf8);
                if (sys::fs::is_regular_file(besideReferencing))
                {
                    includeBuf = sourceMgr.OpenIncludeFile(besideReferencing.str().str(), actualFilePath);
                }
            }

            if (!includeBuf)
            {
                includeBuf = sourceMgr.OpenIncludeFile(includeFileNameUtf8, actualFilePath);
            }

            if (!includeBuf)
            {
                mlir::Location referenceLocation = location;
                if (reference)
                {
                    auto start = reference->pos.textPos > 0 ? reference->pos.textPos : reference->pos.pos;
                    referenceLocation = loc2(referencing, convertWideToUTF8(referencing->fileName), start,
                                             static_cast<int>(reference->_end) - start);
                }

                emitError(referenceLocation, "can't open file: ") << includeFileNameUtf8;
                anyMissing = true;
                return;
            }

            auto includeFilePath = canonicalFilePath(actualFilePath);
            if (!seenFiles.insert(includeFilePath).second)
            {
                return;
            }

            auto id = sourceMgr.AddNewSourceBuffer(std::move(*includeBuf), SMLoc());

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
            includeFile->resolvedPath = convertUTF8toWide(includeFilePath);

            // the directory of the file opened, which is where its own references are
            auto includeFileDir = sys::path::parent_path(actualFilePath).str();
            for (auto &refFile : includeFile->referencedFiles)
            {
                loadReferencedFile(refFile.fileName, includeFile, &refFile, includeFileDir);
            }

            includeFiles.push_back(includeFile);
        };

        for (auto &fileToProcess : filesToProcess)
        {
            loadReferencedFile(fileToProcess, sourceFile, nullptr, StringRef());
        }

        auto sourceFileDir = sys::path::parent_path(fullPath).str();
        for (auto &refFile : sourceFile->referencedFiles)
        {
            loadReferencedFile(refFile.fileName, sourceFile, &refFile, sourceFileDir);
        }

        if (anyMissing)
        {
            return {SourceFile(), {}};
        }

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

        // an imported module importing the program back does not generate the program again
        filesInProgress.insert(fileKey(module));

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
            location = mdi.getCompileUnit(location, "TypeScript Compiler", isOptimized);
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
            auto indexSize = mlir::DataLayoutEntryAttr::get(builder.getIndexType(), builder.getI32IntegerAttr(compileOptions.sizeBits()));
            theModule->setAttr("dlti.dl_spec", mlir::DataLayoutSpecAttr::get(builder.getContext(), {indexSize}));

            // The layout lowering sizes with. LowerToLLVM reads this attribute to build its type
            // converter (pointer sizes, struct and union layout, alignment) and fails without it,
            // so it is required, not decoration; obj.cpp sets the TargetMachine's layout again only
            // after lowering. It also makes --emit=llvm output carry the triple's own layout rather
            // than LLVM's default 64-bit one. Derived from the triple via LLVM's own target
            // registry, using a default TargetOptions, so it matches what obj.cpp later sets for
            // every supported target. On an ABI-name-sensitive backend (ARM, Mips, PowerPC, RISCV)
            // obj.cpp additionally threads a `-target-abi` into TargetOptions, which could change the
            // layout there but not here: lowering's layout would then differ from codegen's. No
            // supported target is affected.
            //
            // --emit=llvm (dump.cpp:dumpLLVMIR) never calls lookupTarget itself, so this is the
            // only place on that path that validates the triple. A silently-swallowed failure
            // here would reproduce the exact bug this task fixes: IR naming a triple but carrying
            // the wrong (default) layout. So treat both failure modes as fatal, consistent with
            // obj.cpp's handling of the same lookup.
            std::string errorMessage;
            llvm::Triple targetTriple(compileOptions.moduleTargetTriple);
            auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
            if (!target)
            {
                emitError(location, "unable to find target for triple '")
                    << compileOptions.moduleTargetTriple << "': " << errorMessage;
                return mlir::failure();
            }

            std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
                targetTriple, "generic", "", llvm::TargetOptions(), std::nullopt));
            if (!machine)
            {
                emitError(location, "unable to create target machine for triple '")
                    << compileOptions.moduleTargetTriple << "'";
                return mlir::failure();
            }

            auto dataLayout = machine->createDataLayout();

            // MLIRGen sizes things with the arch's pointer width (TargetInfo/sizeBits) and lowering
            // with this layout. For ABI-by-environment triples (gnux32, gnuabin32, ilp32) the arch is
            // 64-bit but the ABI's pointers are 32-bit, so the two disagree and nothing downstream can
            // reconcile them. Treat that the same as the lookup failures above: fatal, and naming the
            // triple.
            if (dataLayout.getPointerSizeInBits(0) != static_cast<unsigned>(compileOptions.sizeBits()))
            {
                emitError(location, "target triple '")
                    << compileOptions.moduleTargetTriple << "' has " << dataLayout.getPointerSizeInBits(0)
                    << "-bit pointers but a " << compileOptions.sizeBits()
                    << "-bit architecture; this ABI is not supported";
                return mlir::failure();
            }

            theModule->setAttr(
                mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                builder.getStringAttr(dataLayout.getStringRepresentation()));
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
        varName.append(moduleSymbolSuffix(mainSourceFileName));
        
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

    // Records which memory model this module was built under, so an importer can tell whether
    // objects arriving from it are managed the same way its own are. Emitted alongside the
    // declaration text and under the same condition: a module that exports no declarations
    // cannot be imported, so there is no boundary to mark.
    mlir::LogicalResult MLIRGenImpl::createMemoryModelExportGlobalVar(const GenContext &genContext)
    {
        if (!declExports.rdbuf()->in_avail() || !compileOptions.embedExportDeclarations)
        {
            return mlir::success();
        }

        auto modelName = std::string(memoryModelName(compileOptions.memoryModel));

        auto typeWithInit = [&](mlir::Location location, const GenContext &genContext) {
            auto litValue = V(mlirGenStringValue(location, modelName, true));
            return std::make_tuple(litValue.getType(), litValue, TypeProvided::No);
        };

        auto loc = mlir::UnknownLoc::get(builder.getContext());

        VariableClass varClass = VariableType::Var;
        varClass.isExport = true;
        varClass.isPublic = true;

        // the model is part of the symbol name, so reading it back is a symbol enumeration
        // rather than a data load
        std::string varName(SHARED_LIB_MEMORY_MODEL);
        varName.append(modelName);
        varName.append("_");
        varName.append(moduleSymbolSuffix(mainSourceFileName));

        auto varNameRef = StringRef(varName).copy(stringAllocator);

        registerVariable(loc, varNameRef, true, varClass, typeWithInit, genContext);

        return mlir::success();
    }

    // The names in a library's SHARED_LIB_OWN_FACTS text, one per line, added to the ones this
    // module already has from other libraries.
    void MLIRGenImpl::addImportedOwnNoDrops(StringRef factsText)
    {
        llvm::SmallVector<mlir::Attribute> names;
        llvm::StringSet<> seen;
        if (auto existing = theModule->getAttrOfType<mlir::ArrayAttr>(SHARED_LIB_OWN_NO_DROPS_ATTR_NAME))
        {
            for (auto name : existing.getAsRange<mlir::StringAttr>())
            {
                seen.insert(name.getValue());
                names.push_back(name);
            }
        }

        llvm::SmallVector<StringRef> lines;
        factsText.split(lines, '\n', -1, false);
        for (auto line : lines)
        {
            line = line.trim();
            if (!line.empty() && seen.insert(line).second)
            {
                names.push_back(builder.getStringAttr(line));
            }
        }

        if (!names.empty())
        {
            theModule->setAttr(SHARED_LIB_OWN_NO_DROPS_ATTR_NAME, builder.getArrayAttr(names));
        }
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
        varName.append(moduleSymbolSuffix(mainSourceFileName));

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

    // Whether anything at the root runs when the program starts - code, or a variable whose
    // initializer does. Deliberately a wider question than hasGlobalCode: that one asks only
    // whether an entry function has to be built to hold statements held back from the module
    // level, and answering it "yes" for variables moves them out of the module scope where the
    // file's own functions have to be able to see them.
    bool MLIRGenImpl::hasGlobalInitialization(NodeArray<Statement> statements) {
        for (auto &statement : statements)
        {
            if (isCodeStatment(statement) || statement == SyntaxKind::VariableStatement)
            {
                return true;
            }
        }

        return false;
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
                          bool hasDeferredStatements, const GenContext &genContext)
    {
        // create function
        //auto name = MLIRHelper::getAnonymousName(location, ".main", "");
        auto useGlobalCtor = false;
        std::string name = MAIN_ENTRY_NAME;
        auto fullGlobalFuncName = getFullNamespaceName(name);

        if (theModule.lookupSymbol(fullGlobalFuncName))
        {
            // a user-written `main` already is the entry point, so with nothing deferred to run
            // ahead of it there is nothing left to generate
            if (!hasDeferredStatements)
            {
                return mlir::success();
            }

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
                // nothing was held back from the module level, so this is an empty entry point
                // that exists only to be the program's entry (see the call site)
                if (!hasDeferredStatements)
                {
                    return mlir::success();
                }

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
        // print errors, or hand them to the importer of this file
        if (notResolved && importDiagnostics)
        {
            for (auto &diag : postponedMessages)
            {
                importDiagnostics->push_back(std::move(diag));
            }
        }
        else if (notResolved)
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
            if (emittedFiles.contains(fileKey(includeFile)))
            {
                continue;
            }

            SourceFileScope sourceFileScope(*this, includeFile);

            if (failed(mlirGen(includeFile->statements, genContextPartial)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }

            emittedFiles.insert(fileKey(includeFile));
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

        // a file the program, or another import, already emitted into this module is skipped
        for (auto includeFile : includeFiles)
        {
            if (emittedFiles.contains(fileKey(includeFile)))
            {
                continue;
            }

            SourceFileScope sourceFileScope(*this, includeFile);

            if (failed(mlirGen(includeFile->statements, genContext)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }

            emittedFiles.insert(fileKey(includeFile));
        }

        auto anyGlobalCode = hasGlobalCode(module->statements);
        auto notResolved = processStatements(module->statements, genContext, isMain && anyGlobalCode);       
        if (failed(outputDiagnostics(postponedMessages, notResolved)))
        {
            return mlir::failure();
        }
       
        if (isMain && notResolved == 0)
        {
            // generate code to run at global entry.
            //
            // A program still needs `main` when the root holds no code to defer into it: root-level
            // variables initialize from the global constructors either way, but with no `main` there
            // is nothing for the JIT to call and nothing for the CRT to link against, which is how
            // `class S {} const s = new S();` used to fail with "Symbols not found: [ main ]".
            //
            // A root that only declares things gets no entry point, because that is what a library
            // looks like and its object is linked next to a program that has a `main` of its own -
            // emitting one here is a duplicate symbol at link time.
            //
            // `isExecutable` alone is not the test: it is set only by `--emit=exe`, while everything
            // that links a program compiles with `--emit=obj` and drives the linker itself (same trap
            // as giveEntryPointAnExitCode in LowerToLLVM.cpp). But `--emit=obj` compiles the libraries
            // too, and a library root initializing a variable looks exactly like a program root doing
            // the same, so the object path has to be told which file is the program - that is what
            // generateEntryPoint carries. Guessing it from the emit action instead put a `main` in
            // every library object, and two of those failed to link.
            auto needsEntryPoint = compileOptions.generateEntryPoint && hasGlobalInitialization(module->statements);
            if ((anyGlobalCode || needsEntryPoint) && mlir::failed(
                generateGlobalEntryCode(loc(module), module->statements, anyGlobalCode, genContext)))
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

            if (mlir::failed(createMemoryModelExportGlobalVar(genContext))) {
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
        auto fullPath = includeFilePath(filePath);
        auto canonicalPath = canonicalFilePath(fullPath);

        // Already in this module - referenced with `/// <reference path>`, or imported directly
        // and through another module - so its declarations are all there. Or an import cycle back
        // to a file still being generated: generating it again would never end; what the cycle
        // needs from it and it has not declared yet stays unresolved.
        if (emittedFiles.contains(canonicalPath))
        {
            return mlir::success();
        }

        if (filesInProgress.contains(canonicalPath))
        {
            compileOptions.importCycle = true;
            return mlir::success();
        }

        // Declared by a library already imported: one built from several modules carries each
        // one's declarations (mlirGenImportSharedLib), this one's among them.
        auto suffix = moduleSymbolSuffix(fullPath);
        auto declSymbol = std::string(SHARED_LIB_DECLARATIONS_2UNDERSCORE "_") + suffix;
        auto genericDeclSymbol = std::string(SHARED_LIB_DECLARATIONS_2UNDERSCORE "_generic_") + suffix;
        if (emittedFiles.contains(declSymbol) || emittedFiles.contains(genericDeclSymbol))
        {
            return mlir::success();
        }

        filesInProgress.insert(canonicalPath);
        auto inProgress = llvm::make_scope_exit([&]() { filesInProgress.erase(canonicalPath); });

        // An import that points to a .ts file: compiled, it is a declaration - the module is
        // compiled into an object of its own, and the objects are linked into one program. The
        // JIT does the same with its cache (the module's object is loaded beside the program's);
        // without it there is no other object, so it is included with its bodies, as a
        // referenced file is. (An import that points to a library is mlirGenImportSharedLib.) A
        // .d.ts is read as declarations either way.
        MLIRValueGuard<bool> vg(declarationMode);
        declarationMode = !compileOptions.isJit || compileOptions.jitCache;

        // What the imported module exports is its own to export: it is compiled separately, with
        // its declarations in its own __decls. Added to this module's too, a library holding both
        // (test-runner -shared links every module into one) declared them twice in one import.
        // It also must not reset what this module has collected (mlirCodeGenModule starts a fresh
        // declExports). So the export state is set aside and put back; a type the import
        // declared is added again if one of this module's own exports depends on it.
        auto savedDeclExports = declExports.str();
        auto savedGenericDeclExports = genericDeclExports.str();
        auto savedExportedTypes = exportedTypes;
        auto savedExportCheckedDependenciesTypes = exportCheckedDependenciesTypes;
        auto restoreExports = llvm::make_scope_exit([&]() {
            declExports.str("");
            declExports.clear();
            declExports << savedDeclExports;
            genericDeclExports.str("");
            genericDeclExports.clear();
            genericDeclExports << savedGenericDeclExports;
            exportedTypes = savedExportedTypes;
            exportCheckedDependenciesTypes = savedExportCheckedDependenciesTypes;
        });

        auto [importSource, importIncludeFiles] = loadIncludeFile(location, fullPath);
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

        // An import cycle (a imports b imports a) fails b on its first attempt: a has not declared
        // what b needs yet. a's processStatements tries the import again once it has, so b's errors
        // are the importer's to report, and only if the import still fails.
        mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> diagnostics;
        auto generated = false;
        {
            MLIRValueGuard<mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> *> diagnosticsGuard(importDiagnostics);
            importDiagnostics = &diagnostics;

            generated = mlir::succeeded(mlirDiscoverAllDependencies(importSource, importIncludeFiles)) &&
                mlir::succeeded(mlirCodeGenModule(importSource, importIncludeFiles, false, false));
        }

        // the import's own diagnostic handlers are gone: these reach the importer's
        for (auto &diag : diagnostics)
        {
            builder.getContext()->getDiagEngine().emit(std::move(*diag));
        }

        if (generated)
        {
            // only now: an import that failed is tried again on the next pass, and must fail
            // again rather than find itself already done. Its library declarations, if a
            // library holding it is imported later, are then not declared again either.
            emittedFiles.insert(canonicalPath);
            emittedFiles.insert(declSymbol);
            emittedFiles.insert(genericDeclSymbol);

            // the object the JIT has to load for it; after the ones it imports, which were
            // generated inside it. A pass generated again (discovery, then the module) finds it
            // listed already.
            if (!StringRef(canonicalPath).ends_with(".d.ts") &&
                !llvm::is_contained(compileOptions.sourceImports, canonicalPath))
            {
                compileOptions.sourceImports.push_back(canonicalPath);
            }

            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenImportSharedLib(mlir::Location location, StringRef filePath, bool dynamic, const GenContext &genContext)
    {
        // Imported into this module already - directly and through another module, say: it is
        // loaded, and its declarations are here. A second copy of them redefines every symbol.
        auto canonicalPath = canonicalFilePath(filePath);
        if (emittedFiles.contains(canonicalPath))
        {
            return mlir::success();
        }

        // A library built for another architecture or OS (an i686 DLL for this x64 compiler)
        // cannot be loaded into this process. Its declarations are then read from the file; the
        // program still loads it at run time, through the global constructor below.
        auto loadIntoCompiler = compileOptions.targetInfo.supportsInProcessJit;

        // TODO: ...
        llvm::sys::DynamicLibrary dynLib;
        if (loadIntoCompiler)
        {
            std::string errMsg;
            dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(filePath.str().c_str(), &errMsg);
            if (!dynLib.isValid())
            {
                emitError(location, errMsg);
                return mlir::failure();
            }
        }
        else
        {
            // Only a PE image's declarations can be read from the file (Dump::readExportedCString).
            auto machine = Dump::coffMachine(filePath);
            if (machine == 0)
            {
                emitError(location) << "cannot read declarations from '" << filePath
                                    << "': for a target other than the host, only PE DLLs can be imported";
                return mlir::failure();
            }

            // A DLL for another machine would link, and then not load in the program.
            uint16_t expected = 0;
            llvm::Triple targetTriple(compileOptions.moduleTargetTriple);
            switch (targetTriple.getArch())
            {
            case llvm::Triple::x86:
                expected = llvm::COFF::IMAGE_FILE_MACHINE_I386;
                break;
            case llvm::Triple::x86_64:
                expected = llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
                break;
            case llvm::Triple::aarch64:
                expected = llvm::COFF::IMAGE_FILE_MACHINE_ARM64;
                break;
            default:
                break;
            }

            if (machine != expected)
            {
                emitError(location) << "shared library '" << filePath << "' is built for "
                                    << Dump::coffMachineName(machine) << ", but this program targets "
                                    << (expected ? Dump::coffMachineName(expected)
                                                 : targetTriple.getArchName().str());
                return mlir::failure();
            }
        }

        SmallVector<StringRef> symbols;
        StringRef mlirGctors;
        // every symbol the library exports
        SmallVector<StringRef> symbolsAll;
        // "__tsown_<file>_<hash>", one per module in the library built under own
        SmallVector<StringRef> ownFactsSymbols;
#ifndef GENERATE_IMPORT_INFO_USING_D_TS_FILE
        // loading Binary to get list of symbols
        Dump::getSymbols(filePath, symbolsAll, stringAllocator);

        StringRef memoryModelSymbol;
        for (auto symbol : symbolsAll)
        {
            if (symbol.starts_with(SHARED_LIB_DECLARATIONS_2UNDERSCORE))
            {
                symbols.push_back(symbol);
            }
            else if (symbol.starts_with(SHARED_LIB_MEMORY_MODEL))
            {
                memoryModelSymbol = symbol;
            }
            else if (symbol.starts_with(SHARED_LIB_OWN_FACTS))
            {
                ownFactsSymbols.push_back(symbol);
            }
            else if (symbol == MLIR_GCTORS)
            {
                mlirGctors = symbol;
            }
        }

        // "__tsmm_<model>_<file>_<hash>" - the model is the segment after the prefix. A library
        // with no marker predates it, and everything did collect back then.
        auto libraryModel = std::string("gc");
        if (!memoryModelSymbol.empty())
        {
            auto rest = memoryModelSymbol.drop_front(StringRef(SHARED_LIB_MEMORY_MODEL).size());
            libraryModel = rest.take_until([](char c) { return c == '_'; }).str();
        }

        if (libraryModel != memoryModelName(compileOptions.memoryModel))
        {
            // Allowed on purpose: an object arriving from a module managed differently is
            // treated as immortal rather than rejected, so it leaks instead of being freed
            // twice. See docs/reference-counting-evaluation.md section 4.
            emitWarning(location) << "shared library '" << filePath << "' was built with -mm="
                                  << libraryModel << ", this module with -mm="
                                  << memoryModelName(compileOptions.memoryModel)
                                  << ". Objects crossing between them are never reclaimed.";
        }

        // One collector per process. A gc library that linked Boehm statically brings a second
        // collector into this one, which cannot see this module's references and frees objects it
        // still holds - silently, as wrong values rather than a crash. Windows only: an ELF
        // program exports its collector (exe.cpp), so a shared object's own copy is never called.
        // See docs/single-gc-collector-design.md, step 4.
        if (compileOptions.isWindows && compileOptions.needsGCRuntime() && libraryModel == "gc" &&
            Dump::containsGarbageCollector(filePath))
        {
            emitError(location) << "shared library '" << filePath
                                << "' links its own garbage collector (the static gc.lib). Objects crossing "
                                   "between it and this module can be freed while still in use. Rebuild it "
                                   "with tslang --emit=dll, which links gc.dll.";
            return mlir::failure();
        }
#else
        // only 1 file to load        
        symbols.push_back(SHARED_LIB_DECLARATIONS_2UNDERSCORE);
        if (!loadIntoCompiler)
        {
            // read from the file below, which tells a missing symbol from an unreadable one
            Dump::getSymbols(filePath, symbolsAll, stringAllocator);
        }
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

            // The shared-lib load + symbol resolution call into the runtime's
            // library list (LLVM's sys::DynamicLibrary under the JIT), which uses
            // std::vector. In debug builds STL
            // iterators take a global lock that the CRT only initializes via its
            // own '_Init_locks'/'initlocks' dynamic initializer (in .CRT$XCU).
            // FIRST_GLOBAL_CONSTRUCTOR_PRIORITY (100) places this ctor BEFORE that
            // CRT init -> entering an uninitialized CRITICAL_SECTION -> crash.
            // Use the same band as the per-symbol __cctors (LAST) so it runs after
            // 'initlocks'; it is emitted before them, so it still loads the library
            // before any tslang_search_for_address_of_symbol runs.
            addGlobalConstructor(location, fullInitGlobalFuncName);
        }

        // Generics first: a declaration elsewhere names their specializations (a field typed
        // `Box<Tree>`), and the symbols come sorted, which put __decls_<module> before
        // __decls_generic_<module> - "generic type Box can't be found". Registering a generic
        // only records it, so one naming a class declared after it is fine.
        std::stable_partition(symbols.begin(), symbols.end(), [](StringRef symbol) {
            return symbol.starts_with(std::string(SHARED_LIB_DECLARATIONS_2UNDERSCORE) + "_generic_");
        });

        // A library can hold several modules (test-runner -shared links them all into one), each
        // with its own __decls_<module>. A module already in this module - imported as source,
        // or through another library - is not declared again; see mlirGenInclude.
        SmallVector<StringRef> declaredSymbols;
        for (auto declSymbol : symbols)
        {
            if (emittedFiles.contains(declSymbol))
            {
                continue;
            }

            declaredSymbols.push_back(declSymbol);

            // TODO: for now, we have code in TS to load methods from DLL/Shared libs
            const char *declText = nullptr;
            std::optional<std::string> declTextFromFile;
            if (loadIntoCompiler)
            {
                if (auto addrOfDeclText = dynLib.getAddressOfSymbol(declSymbol.str().c_str()))
                {
                    declText = *(const char**)addrOfDeclText;
                }
            }
            else if (llvm::is_contained(symbolsAll, declSymbol))
            {
                declTextFromFile = Dump::readExportedCString(filePath, declSymbol);
                if (!declTextFromFile)
                {
                    emitError(location) << "shared library '" << filePath << "' exports " << declSymbol
                                        << " but its declarations could not be read from the file";
                    return mlir::failure();
                }

                declText = declTextFromFile->c_str();
            }

            if (declText)
            {
                std::string result;
                // process shared lib declarations
                auto dataPtr = declText;
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

        // What a library built under own says about its functions (SHARED_LIB_OWN_FACTS). Kept in
        // every model, so MLIRGen emits the same under own as under rc; only own reads it.
        for (auto factsSymbol : ownFactsSymbols)
        {
            std::optional<std::string> factsText;
            if (loadIntoCompiler)
            {
                if (auto addrOfFacts = dynLib.getAddressOfSymbol(factsSymbol.str().c_str()))
                {
                    factsText = std::string(*(const char **)addrOfFacts);
                }
            }
            else
            {
                factsText = Dump::readExportedCString(filePath, factsSymbol);
            }

            if (!factsText)
            {
                emitError(location) << "shared library '" << filePath << "' exports " << factsSymbol
                                    << " but it could not be read from the file";
                return mlir::failure();
            }

            addImportedOwnNoDrops(*factsText);
        }

        // only now: an import that failed is tried again on the next pass, and must fail again
        // rather than find itself already done
        emittedFiles.insert(canonicalPath);
        if (!llvm::is_contained(compileOptions.sharedLibraryImports, canonicalPath))
        {
            compileOptions.sharedLibraryImports.push_back(canonicalPath);
        }

        for (auto declSymbol : declaredSymbols)
        {
            emittedFiles.insert(declSymbol);
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
            if (mlir::failed(mlirGenImportSharedLib(location, fullPath, dynamic, genContext)))
            {
                return mlir::failure();
            }
        }
        else if (mlir::failed(mlirGenInclude(location, stringVal, genContext)))
        {
            return mlir::failure();
        }

        return mlirGenImportBindings(importDeclarationAST->importClause);
    }

    // An imported module's declarations are generated at this module's top level, whatever the
    // import names - there is no per-module scope - so `import { a }` and `import './m'` need
    // nothing more. `import { a as b }` makes b another name for a. `import * as M` makes M a
    // namespace of its own with nothing in it: a name looked for in it is then looked for at the
    // top level (resolveIdentifier, resolveTypeByName), which is where the module's declarations
    // are, and M.a is the module's a. Symbol names do not change, so the module's own object
    // file still provides the bodies. M.x also finds a top-level x of another file; code that
    // TypeScript accepts means the same.
    mlir::LogicalResult MLIRGenImpl::mlirGenImportBindings(ImportClause importClause)
    {
        if (!importClause || !importClause->namedBindings)
        {
            return mlir::success();
        }

        auto namedBindings = importClause->namedBindings;
        if (namedBindings == SyntaxKind::NamespaceImport)
        {
            auto namespaceName = MLIRHelper::getName(namedBindings.as<NamespaceImport>()->name, stringAllocator);
            MLIRNamespaceGuard nsGuard(currentNamespace);
            registerNamespace(namespaceName);
            return mlir::success();
        }

        if (namedBindings == SyntaxKind::NamedImports)
        {
            for (auto element : namedBindings.as<NamedImports>()->elements)
            {
                if (!element->propertyName)
                {
                    continue;
                }

                auto target = MLIRHelper::getName(element->propertyName, stringAllocator);
                auto alias = MLIRHelper::getName(element->name, stringAllocator);
                if (alias != target)
                {
                    getImportAliasMap()[alias] = target;
                }
            }
        }

        return mlir::success();
    }

    StringRef MLIRGenImpl::resolveImportAlias(StringRef name)
    {
        return getImportAliasMap().lookup(name);
    }

} // namespace mlirgen
} // namespace typescript
