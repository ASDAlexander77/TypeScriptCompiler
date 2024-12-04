// TODO: it seems in Jit mode, LLVM Engine can resolve external references from loading DLLs

#ifdef GC_ENABLE
#define ADD_GC_ATTRIBUTE true
#endif

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DiagnosticHelper.h"
#include "TypeScript/ObjDumper.h"

#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/MLIRLogic/MLIRGenContext.h"
#include "TypeScript/MLIRLogic/MLIRNamespaceGuard.h"
#include "TypeScript/MLIRLogic/MLIRLocationGuard.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRValueGuard.h"
#include "TypeScript/MLIRLogic/MLIRDebugInfoHelper.h"
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVC.h"
#include "TypeScript/MLIRLogic/MLIRPrinter.h"
#include "TypeScript/MLIRLogic/MLIRDeclarationPrinter.h"
#include "TypeScript/MLIRLogic/TypeOfOpHelper.h"
#include "TypeScript/VisitorAST.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "dump.h"
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Diagnostics.h"
#ifdef ENABLE_ASYNC
#include "mlir/Dialect/Async/IR/Async.h"
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
//#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/WithColor.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <set>

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using DITableScopeT = llvm::ScopedHashTableScope<StringRef, mlir::LLVM::DIScopeAttr>;

// TODO: optimize of amount of calls to detect return types and if it is was calculated before then do not run it all
// the time

SourceMgrDiagnosticHandlerEx::SourceMgrDiagnosticHandlerEx(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) : mlir::SourceMgrDiagnosticHandler(mgr, ctx)
{
}

void SourceMgrDiagnosticHandlerEx::emit(mlir::Diagnostic &diag)
{
    emitDiagnostic(diag);
}

namespace
{

enum class IsGeneric
{
    False,
    True
};

enum class Reason
{
    None,
    FailedConstraint,
    Failure,
    NoConstraint
};

enum class TypeProvided
{
    No,
    Yes
};

enum class DisposeDepth
{
    CurrentScope,
    CurrentScopeKeepAfterUse,
    LoopScope,
    FullStack
};

enum class Stages
{
    Discovering,
    SourceGeneration
};

typedef std::tuple<mlir::Type, mlir::Value, TypeProvided> TypeValueInitType;
typedef std::function<TypeValueInitType(mlir::Location, const GenContext &)> TypeValueInitFuncType;

/// Implementation of a simple MLIR emission from the TypeScript AST.
///
/// This will emit operations that are specific to the TypeScript language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl
{
  public:
    MLIRGenImpl(const mlir::MLIRContext &context, const llvm::StringRef &fileNameParam,
                const llvm::StringRef &pathParam, const llvm::SourceMgr &sourceMgr, CompileOptions &compileOptions)
        : builder(&const_cast<mlir::MLIRContext &>(context)), 
          sourceMgr(const_cast<llvm::SourceMgr &>(sourceMgr)),
          sourceMgrHandler(const_cast<llvm::SourceMgr &>(sourceMgr), &const_cast<mlir::MLIRContext &>(context)),
          mth(&const_cast<mlir::MLIRContext &>(context), 
            std::bind(&MLIRGenImpl::getClassInfoByFullName, this, std::placeholders::_1), 
            std::bind(&MLIRGenImpl::getGenericClassInfoByFullName, this, std::placeholders::_1), 
            std::bind(&MLIRGenImpl::getInterfaceInfoByFullName, this, std::placeholders::_1), 
            std::bind(&MLIRGenImpl::getGenericInterfaceInfoByFullName, this, std::placeholders::_1)),
          compileOptions(compileOptions), 
          mainSourceFileName(fileNameParam),
          path(pathParam),
          declarationMode(false),
          tempEntryBlock(nullptr),
          overwriteLoc(mlir::UnknownLoc::get(builder.getContext()))
    {
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();

        std::vector<std::string> includeDirs;
        includeDirs.push_back(pathParam.str());
        if (!compileOptions.noDefaultLib)
        {
            SmallString<256> defaultLibPath(compileOptions.defaultDeclarationTSFile);
            sys::path::remove_filename(defaultLibPath);    
            includeDirs.push_back(defaultLibPath.str().str());        
        }

        const_cast<llvm::SourceMgr &>(sourceMgr).setIncludeDirs(includeDirs);
    }

    mlir::LogicalResult report(SourceFile module, const std::vector<SourceFile> &includeFiles)
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

    std::pair<SourceFile, std::vector<SourceFile>> loadMainSourceFile()
    {
        const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        auto sourceFileLoc = mlir::FileLineColLoc::get(builder.getContext(),
                    sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
        return loadSourceBuf(sourceFileLoc, sourceBuf, true);
    }    

    std::pair<SourceFile, std::vector<SourceFile>> loadSourceFile(SMLoc loc)
    {
        const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.FindBufferContainingLoc(loc));
        auto sourceFileLoc = mlir::FileLineColLoc::get(builder.getContext(),
                    sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
        return loadSourceBuf(sourceFileLoc, sourceBuf, true);
    }        

    std::pair<SourceFile, std::vector<SourceFile>> loadSourceBuf(mlir::Location location, const llvm::MemoryBuffer *sourceBuf, bool isMain = false)
    {
        std::vector<SourceFile> includeFiles;
        std::vector<string> filesToProcess;

        Parser parser;
        auto sourceFile = parser.parseSourceFile(
            stows(mainSourceFileName.str()), 
            stows(sourceBuf->getBuffer().str()), 
            ScriptTarget::Latest);

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
        }

        for (auto refFile : sourceFile->referencedFiles)
        {
            filesToProcess.push_back(refFile.fileName);
        }

        while (filesToProcess.size() > 0)
        {
            string includeFileName = filesToProcess.back();
            SmallString<256> fullPath;
            auto includeFileNameUtf8 = convertWideToUTF8(includeFileName);
            sys::path::append(fullPath, includeFileNameUtf8);

            filesToProcess.pop_back();

            std::string actualFilePath;
            auto id = sourceMgr.AddIncludeFile(std::string(fullPath), SMLoc(), actualFilePath);
            if (!id)
            {
                emitError(location, "can't open file: ") << fullPath;
                continue;
            }

            const auto *sourceBuf = sourceMgr.getMemoryBuffer(id);

            Parser parser;
            auto includeFile =
                parser.parseSourceFile(convertUTF8toWide(actualFilePath), stows(sourceBuf->getBuffer().str()), ScriptTarget::Latest);
            for (auto refFile : includeFile->referencedFiles)
            {
                filesToProcess.push_back(refFile.fileName);
            }

            includeFiles.push_back(includeFile);
        }

        std::reverse(includeFiles.begin(), includeFiles.end());

        return {sourceFile, includeFiles};
    }

    mlir::LogicalResult showMessages(SourceFile module, std::vector<SourceFile> includeFiles)
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

    mlir::ModuleOp mlirGenSourceFile(SourceFile module, std::vector<SourceFile> includeFiles)
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

  private:
    mlir::LogicalResult mlirGenCodeGenInit(SourceFile module)
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
    /// Create a dependency declaration file for `--emit=dll` option.
    ///
    mlir::LogicalResult createDependencyDeclarationFile(StringRef outputFilename,
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

    mlir::LogicalResult createDeclarationExportGlobalVar(const GenContext &genContext)
    {
        if (!declExports.rdbuf()->in_avail())
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

    bool isCodeStatment(SyntaxKind kind)
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

    int processStatements(NodeArray<Statement> statements,
                          const GenContext &genContext,
                          bool isRoot = false)
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

    bool hasGlobalCode(NodeArray<Statement> statements) {
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

    mlir::LogicalResult generateGlobalEntryCode(mlir::Location location, NodeArray<Statement> statements,
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
                            variableStatement->declarationList->flags &= ~NodeFlags::Const;                        
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
            auto parentModule = theModule;
            MLIRCodeLogicHelper mclh(builder, location);

            builder.setInsertionPointToStart(parentModule.getBody());
            mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());            

            // priority is lowest to load as first dependencies
            builder.create<mlir_ts::GlobalConstructorOp>(
                location, 
                mlir::FlatSymbolRefAttr::get(builder.getContext(), 
                fullGlobalFuncName), 
                builder.getIndexAttr(LAST_GLOBAL_CONSTRUCTOR_PRIORITY));            
        }
        
        return mlir::success();
    }

    mlir::LogicalResult outputDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages,
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

    mlir::LogicalResult mlirDiscoverAllDependencies(SourceFile module, std::vector<SourceFile> includeFiles = {})
    {
        mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedMessages;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
            postponedMessages.emplace_back(new mlir::Diagnostic(std::move(diag)));
        });

        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(
            fullNameGlobalsMap);

        // Process of discovery here
        GenContext genContextPartial{};
        genContextPartial.allowPartialResolve = true;
        genContextPartial.dummyRun = true;
        genContextPartial.rootContext = &genContextPartial;
        genContextPartial.postponedMessages = &postponedMessages;
        // TODO: no need to clean up here as whole module will be removed
        //genContextPartial.cleanUps = new mlir::SmallVector<mlir::Block *>();
        //genContextPartial.cleanUpOps = new mlir::SmallVector<mlir::Operation *>();

        for (auto includeFile : includeFiles)
        {
            MLIRValueGuard<llvm::StringRef> vgFileName(mainSourceFileName); 
            auto fileNameUtf8 = convertWideToUTF8(includeFile->fileName);
            mainSourceFileName = fileNameUtf8;

            MLIRValueGuard<ts::SourceFile> vgSourceFile(sourceFile);
            sourceFile = includeFile;

            if (failed(mlirGen(includeFile->statements, genContextPartial)))
            {
                outputDiagnostics(postponedMessages, 1);
                return mlir::failure();
            }
        }

        auto notResolved = processStatements(module->statements, genContextPartial);

        genContextPartial.clean();

        // clean up
        clearTempModule();
        theModule.getBody()->clear();

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

    mlir::LogicalResult mlirCodeGenModule(SourceFile module, std::vector<SourceFile> includeFiles = {},
                                          bool validate = true, bool isMain = true)
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
            MLIRValueGuard<llvm::StringRef> vgFileName(mainSourceFileName); 
            auto fileNameUtf8 = convertWideToUTF8(includeFile->fileName);
            mainSourceFileName = fileNameUtf8;

            MLIRValueGuard<ts::SourceFile> vgSourceFile(sourceFile);
            sourceFile = includeFile;

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

    bool registerNamespace(llvm::StringRef namePtr, bool isFunctionNamespace = false)
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

    mlir::LogicalResult exitNamespace()
    {
        // TODO: it will increase reference count, investigate how to fix it
        currentNamespace = currentNamespace->parentNamespace;
        return mlir::success();
    }

    mlir::LogicalResult mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
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

    mlir::LogicalResult mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
#ifdef MODULE_AS_NAMESPACE
        return mlirGenNamespace(moduleDeclarationAST, genContext);
#else
        auto isNamespace = (moduleDeclarationAST->flags & NodeFlags::Namespace) == NodeFlags::Namespace;
        auto isNestedNamespace =
            (moduleDeclarationAST->flags & NodeFlags::NestedNamespace) == NodeFlags::NestedNamespace;
        if (isNamespace || isNestedNamespace)
        {
            return mlirGenNamespace(moduleDeclarationAST, genContext);
        }

        auto location = loc(moduleDeclarationAST);

        auto moduleName = MLIRHelper::getName(moduleDeclarationAST->name);

        auto moduleOp = builder.create<mlir::ModuleOp>(location, StringRef(moduleName));

        builder.setInsertionPointToStart(&moduleOp.getBody().front());

        // save module theModule
        auto parentModule = theModule;
        theModule = moduleOp;

        GenContext moduleGenContext{};
        auto result = mlirGenBody(moduleDeclarationAST->body, moduleGenContext);
        auto result = V(result);

        // restore
        theModule = parentModule;

        builder.setInsertionPointAfter(moduleOp);

        return result;
#endif
    }

    mlir::LogicalResult mlirGenInclude(mlir::Location location, StringRef filePath, const GenContext &genContext)
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

        if (mlir::succeeded(mlirDiscoverAllDependencies(importSource, importIncludeFiles)) &&
            mlir::succeeded(mlirCodeGenModule(importSource, importIncludeFiles, false, false)))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenImportSharedLib(mlir::Location location, StringRef filePath, bool dynamic, const GenContext &genContext)
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

            auto parentModule = theModule;
            MLIRCodeLogicHelper mclh(builder, location);

            builder.setInsertionPointToStart(parentModule.getBody());
            mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());            

            // priority is lowest to load as first dependencies
            builder.create<mlir_ts::GlobalConstructorOp>(
                location, mlir::FlatSymbolRefAttr::get(builder.getContext(), fullInitGlobalFuncName), builder.getIndexAttr(FIRST_GLOBAL_CONSTRUCTOR_PRIORITY));
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

                    auto importData = convertUTF8toWide(dataPtr);
                    if (mlir::failed(parsePartialStatements(importData, genContext, false, true)))
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

    mlir::LogicalResult mlirGen(ImportDeclaration importDeclarationAST, const GenContext &genContext)
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

    boolean isStatement(SyntaxKind kind)
    {
        switch (kind)
        {
        case SyntaxKind::FunctionDeclaration:
        case SyntaxKind::ExpressionStatement:
        case SyntaxKind::VariableStatement:
        case SyntaxKind::IfStatement:
        case SyntaxKind::ReturnStatement:
        case SyntaxKind::LabeledStatement:
        case SyntaxKind::DoStatement:
        case SyntaxKind::WhileStatement:
        case SyntaxKind::ForStatement:
        case SyntaxKind::ForInStatement:
        case SyntaxKind::ForOfStatement:
        case SyntaxKind::ContinueStatement:
        case SyntaxKind::BreakStatement:
        case SyntaxKind::SwitchStatement:
        case SyntaxKind::ThrowStatement:
        case SyntaxKind::TryStatement:
        case SyntaxKind::TypeAliasDeclaration:
        case SyntaxKind::Block:
        case SyntaxKind::EnumDeclaration:
        case SyntaxKind::ClassDeclaration:
        case SyntaxKind::InterfaceDeclaration:
        case SyntaxKind::ImportEqualsDeclaration:
        case SyntaxKind::ImportDeclaration:
        case SyntaxKind::ModuleDeclaration:
        case SyntaxKind::DebuggerStatement:
        case SyntaxKind::EmptyStatement:
            return true;
        default:
            return false;
        }
    }

    boolean isExpression(SyntaxKind kind)
    {
        switch (kind)       
        {
        case SyntaxKind::Identifier:
        case SyntaxKind::PropertyAccessExpression:
        case SyntaxKind::CallExpression:
        case SyntaxKind::NumericLiteral:
        case SyntaxKind::StringLiteral:
        case SyntaxKind::NoSubstitutionTemplateLiteral:
        case SyntaxKind::BigIntLiteral:
        case SyntaxKind::NullKeyword:
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
        case SyntaxKind::ArrayLiteralExpression:
        case SyntaxKind::ObjectLiteralExpression:
        case SyntaxKind::SpreadElement:
        case SyntaxKind::BinaryExpression:
        case SyntaxKind::PrefixUnaryExpression:
        case SyntaxKind::PostfixUnaryExpression:
        case SyntaxKind::ParenthesizedExpression:
        case SyntaxKind::TypeOfExpression:
        case SyntaxKind::ConditionalExpression:
        case SyntaxKind::ElementAccessExpression:
        case SyntaxKind::FunctionExpression:
        case SyntaxKind::ArrowFunction:
        case SyntaxKind::TypeAssertionExpression:
        case SyntaxKind::AsExpression:
        case SyntaxKind::TemplateExpression:
        case SyntaxKind::TaggedTemplateExpression:
        case SyntaxKind::NewExpression:
        case SyntaxKind::DeleteExpression:
        case SyntaxKind::ThisKeyword:
        case SyntaxKind::SuperKeyword:
        case SyntaxKind::VoidExpression:
        case SyntaxKind::YieldExpression:
        case SyntaxKind::AwaitExpression:
        case SyntaxKind::NonNullExpression:
        case SyntaxKind::ClassExpression:
        case SyntaxKind::OmittedExpression:
        case SyntaxKind::ExpressionWithTypeArguments:
            return true;
        default:
            return false;
        }        
    }

    mlir::LogicalResult mlirGenBody(Node body, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)body;
        if (kind == SyntaxKind::Block)
        {
            return mlirGen(body.as<ts::Block>(), genContext);
        }

        if (kind == SyntaxKind::ModuleBlock)
        {
            return mlirGen(body.as<ModuleBlock>(), genContext);
        }

        if (isStatement(body))
        {
            return mlirGen(body.as<Statement>(), genContext);
        }

        if (isExpression(body))
        {
            auto result = mlirGen(body.as<Expression>(), genContext);
            EXIT_IF_FAILED(result)
            auto resultValue = V(result);
            if (resultValue)
            {
                return mlirGenReturnValue(loc(body), resultValue, false, genContext);
            }

            builder.create<mlir_ts::ReturnOp>(loc(body));
            return mlir::success();
        }

        llvm_unreachable("unknown body type");
    }

    void clearState(NodeArray<Statement> statements)
    {
        for (auto &statement : statements)
        {
            statement->processed = false;
        }
    }

    mlir::LogicalResult mlirGen(NodeArray<Statement> statements, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        clearState(statements);

        auto notResolved = 0;
        do
        {
            auto noErrorLocation = true;
            mlir::Location errorLocation = mlir::UnknownLoc::get(builder.getContext());
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    if (noErrorLocation)
                    {
                        errorLocation = loc(statement);
                        noErrorLocation = false;
                    }

                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depends on other class declarations
                emitError(errorLocation, "can't resolve dependencies in namespace");
                return mlir::failure();
            }
        } while (notResolved > 0);

        // clear states to be able to run second time
        clearState(statements);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(
        NodeArray<Statement> statements, 
        std::function<bool(Statement)> filter,
        int& processedStatements,
        const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        clearState(statements);

        auto notResolved = 0;
        do
        {
            auto noErrorLocation = true;
            mlir::Location errorLocation = mlir::UnknownLoc::get(builder.getContext());
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (!filter(statement))
                {
                    continue;
                }

                // clear previous errors
                genContext.postponedMessages->clear();
                if (failed(mlirGen(statement, genContext)))
                {
                    if (noErrorLocation)
                    {
                        errorLocation = loc(statement);
                        noErrorLocation = false;
                    }

                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                    processedStatements++;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depends on other class declarations
                emitError(errorLocation, "can't resolve dependencies in namespace");
                return mlir::failure();
            }
        } while (notResolved > 0);

        // clear states to be able to run second time
        clearState(statements);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ModuleBlock moduleBlockAST, const GenContext &genContext)
    {
        return mlirGen(moduleBlockAST->statements, genContext);
    }

    static bool processIfDeclaration(Statement statement)
    {
        switch ((SyntaxKind)statement)
        {
        case SyntaxKind::FunctionDeclaration:
        case SyntaxKind::ClassDeclaration:
        case SyntaxKind::InterfaceDeclaration:
        case SyntaxKind::EnumDeclaration:
            return true;
        }

        return false;
    }

    mlir::LogicalResult mlirGen(ts::Block blockAST, const GenContext &genContext, int skipStatements = 0)
    {
        auto location = loc(blockAST);

        SymbolTableScopeT varScope(symbolTable);
        GenContext genContextUsing(genContext);
        genContextUsing.parentBlockContext = &genContext;

        DITableScopeT debugBlockScope(debugScope);
        if (compileOptions.generateDebugInfo && !blockAST->parent)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            mdi.setLexicalBlock(location);
        }

        auto usingVars = std::make_unique<SmallVector<ts::VariableDeclarationDOM::TypePtr>>();
        genContextUsing.usingVars = usingVars.get();

        EXIT_IF_FAILED(mlirGenNoScopeVarsAndDisposable(blockAST, genContextUsing, skipStatements));

        // we need to call dispose for those which are in "using"
        // default value for genContext.cleanUpUsingVarsFlag = CurrentScope
        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScope, {}, &genContextUsing));

        return mlir::success();
    }

    mlir::LogicalResult mlirGenNoScopeVarsAndDisposable(ts::Block blockAST, const GenContext &genContext, int skipStatements = 0)
    {
        auto location = loc(blockAST);

        if (genContext.generatedStatements.size() > 0)
        {
            // we need to process it only once (to prevent it processing in nested functions with body)
            NodeArray<Statement> generatedStatements;
            std::copy(genContext.generatedStatements.begin(), genContext.generatedStatements.end(),
                      std::back_inserter(generatedStatements));

            // clean up
            const_cast<GenContext&>(genContext).generatedStatements.clear();

            // auto generated code
            for (auto statement : generatedStatements)
            {
                if (failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        // clear states to be able to run second time
        // for generic methods/types
        clearState(blockAST->statements);

        for (auto statement : blockAST->statements)
        {
            if (skipStatements-- > 0) 
            {
                continue;
            }

            if (statement->processed)
            {
                continue;
            }

            // TODO: we have issue, we can create IfStatement/ForStatment/WhileStatment (etc) which have blocks
            // which will not be removed as it is partially process code
            // so it will not be removed and cause "dirt" in the code which wil cause compile issue
            if (failed(mlirGen(statement, genContext)))
            {
                // special case to show errors in case of discovery, generics & evaluates
                if (genContext.isStopped())
                {
                    return mlir::failure();
                }

                // now try to process all internal declarations
                // process all declrations
                auto processedDeclStatements = 0;
                if (mlir::failed(mlirGen(blockAST->statements, processIfDeclaration, processedDeclStatements, genContext)))
                {
                    return mlir::failure();
                }

                // try to process it again
                if (processedDeclStatements == 0 || failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            statement->processed = true;
        }

        // clear states to be able to run second time
        clearState(blockAST->statements);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenDisposable(mlir::Location location, DisposeDepth disposeDepth, std::string loopLabel, const GenContext* genContext)
    {
        if (genContext->usingVars != nullptr)
        {
            for (auto vi : *genContext->usingVars)
            {
                auto varInTable = symbolTable.lookup(vi->getName());
                if (!varInTable.first)
                {
                    llvm_unreachable("can't find local variable");
                }

                auto callResult = mlirGenCallThisMethod(location, varInTable.first, SYMBOL_DISPOSE, undefined, {}, *genContext);
                EXIT_IF_FAILED(callResult);            
            }

            // remove when used
            if (disposeDepth == DisposeDepth::CurrentScope)
            {
                const_cast<GenContext *>(genContext)->usingVars = nullptr;
            }

            auto continueIntoDepth = disposeDepth == DisposeDepth::FullStack
                    || disposeDepth == DisposeDepth::LoopScope && genContext->isLoop && genContext->loopLabel != loopLabel;
            if (continueIntoDepth)
            {
                EXIT_IF_FAILED(mlirGenDisposable(location, disposeDepth, {}, genContext->parentBlockContext));
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(Statement statementAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)statementAST;
        if (kind == SyntaxKind::FunctionDeclaration)
        {
            return mlirGen(statementAST.as<FunctionDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ExpressionStatement)
        {
            return mlirGen(statementAST.as<ExpressionStatement>(), genContext);
        }
        else if (kind == SyntaxKind::VariableStatement)
        {
            return mlirGen(statementAST.as<VariableStatement>(), genContext);
        }
        else if (kind == SyntaxKind::IfStatement)
        {
            return mlirGen(statementAST.as<IfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ReturnStatement)
        {
            return mlirGen(statementAST.as<ReturnStatement>(), genContext);
        }
        else if (kind == SyntaxKind::LabeledStatement)
        {
            return mlirGen(statementAST.as<LabeledStatement>(), genContext);
        }
        else if (kind == SyntaxKind::DoStatement)
        {
            return mlirGen(statementAST.as<DoStatement>(), genContext);
        }
        else if (kind == SyntaxKind::WhileStatement)
        {
            return mlirGen(statementAST.as<WhileStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForStatement)
        {
            return mlirGen(statementAST.as<ForStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForInStatement)
        {
            return mlirGen(statementAST.as<ForInStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForOfStatement)
        {
            return mlirGen(statementAST.as<ForOfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ContinueStatement)
        {
            return mlirGen(statementAST.as<ContinueStatement>(), genContext);
        }
        else if (kind == SyntaxKind::BreakStatement)
        {
            return mlirGen(statementAST.as<BreakStatement>(), genContext);
        }
        else if (kind == SyntaxKind::SwitchStatement)
        {
            return mlirGen(statementAST.as<SwitchStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ThrowStatement)
        {
            return mlirGen(statementAST.as<ThrowStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TryStatement)
        {
            return mlirGen(statementAST.as<TryStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAliasDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<TypeAliasDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::Block)
        {
            return mlirGen(statementAST.as<ts::Block>(), genContext);
        }
        else if (kind == SyntaxKind::EnumDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<EnumDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ClassDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ClassDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::InterfaceDeclaration) 
        {
            // declaration
            return mlirGen(statementAST.as<InterfaceDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ImportEqualsDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportEqualsDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ImportDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ModuleDeclaration)
        {
            return mlirGen(statementAST.as<ModuleDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::DebuggerStatement)
        {
            return mlirGen(statementAST.as<DebuggerStatement>(), genContext);
        }
        else if (kind == SyntaxKind::EmptyStatement ||
                 kind == SyntaxKind::Unknown /*TODO: temp solution to treat null statements as empty*/)
        {
            return mlir::success();
        }

        llvm_unreachable("unknown statement type");
    }

    mlir::LogicalResult mlirGen(ExpressionStatement expressionStatementAST, const GenContext &genContext)
    {
        auto result = mlirGen(expressionStatementAST->expression, genContext);
        EXIT_IF_FAILED(result)
        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(Expression expressionAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)expressionAST;
        if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(expressionAST.as<Identifier>(), genContext);
        }
        else if (kind == SyntaxKind::PropertyAccessExpression)
        {
            return mlirGen(expressionAST.as<PropertyAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::CallExpression)
        {
            return mlirGen(expressionAST.as<CallExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NumericLiteral)
        {
            return mlirGen(expressionAST.as<NumericLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::StringLiteral)
        {
            return mlirGen(expressionAST.as<ts::StringLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NoSubstitutionTemplateLiteral)
        {
            return mlirGen(expressionAST.as<NoSubstitutionTemplateLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::BigIntLiteral)
        {
            return mlirGen(expressionAST.as<BigIntLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::RegularExpressionLiteral)
        {
            return mlirGen(expressionAST.as<RegularExpressionLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NullKeyword)
        {
            return mlirGen(expressionAST.as<NullLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::TrueKeyword)
        {
            return mlirGen(expressionAST.as<TrueLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::FalseKeyword)
        {
            return mlirGen(expressionAST.as<FalseLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayLiteralExpression)
        {
            return mlirGen(expressionAST.as<ArrayLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectLiteralExpression)
        {
            return mlirGen(expressionAST.as<ObjectLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::SpreadElement)
        {
            return mlirGen(expressionAST.as<SpreadElement>(), genContext);
        }
        else if (kind == SyntaxKind::BinaryExpression)
        {
            return mlirGen(expressionAST.as<BinaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PrefixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PrefixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PostfixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PostfixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedExpression)
        {
            return mlirGen(expressionAST.as<ParenthesizedExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOfExpression)
        {
            return mlirGen(expressionAST.as<TypeOfExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ConditionalExpression)
        {
            return mlirGen(expressionAST.as<ConditionalExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ElementAccessExpression)
        {
            return mlirGen(expressionAST.as<ElementAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::FunctionExpression)
        {
            return mlirGen(expressionAST.as<FunctionExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ArrowFunction)
        {
            return mlirGen(expressionAST.as<ArrowFunction>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAssertionExpression)
        {
            return mlirGen(expressionAST.as<TypeAssertion>(), genContext);
        }
        else if (kind == SyntaxKind::AsExpression)
        {
            return mlirGen(expressionAST.as<AsExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateExpression)
        {
            return mlirGen(expressionAST.as<TemplateLiteralLikeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TaggedTemplateExpression)
        {
            return mlirGen(expressionAST.as<TaggedTemplateExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NewExpression)
        {
            return mlirGen(expressionAST.as<NewExpression>(), genContext);
        }
        else if (kind == SyntaxKind::DeleteExpression)
        {
            mlirGen(expressionAST.as<DeleteExpression>(), genContext);
            return mlir::success();
        }
        else if (kind == SyntaxKind::ThisKeyword)
        {
            if ((expressionAST->internalFlags & InternalFlags::ThisArgAlias) == InternalFlags::ThisArgAlias)
            {
                return mlirGen(loc(expressionAST), THIS_ALIAS, genContext);
            }

            return mlirGen(loc(expressionAST), THIS_NAME, genContext);
        }
        else if (kind == SyntaxKind::SuperKeyword)
        {
            return mlirGen(loc(expressionAST), SUPER_NAME, genContext);
        }
        else if (kind == SyntaxKind::VoidExpression)
        {
            return mlirGen(expressionAST.as<VoidExpression>(), genContext);
        }
        else if (kind == SyntaxKind::YieldExpression)
        {
            return mlirGen(expressionAST.as<YieldExpression>(), genContext);
        }
        else if (kind == SyntaxKind::AwaitExpression)
        {
            return mlirGen(expressionAST.as<AwaitExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NonNullExpression)
        {
            return mlirGen(expressionAST.as<NonNullExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ClassExpression)
        {
            return mlirGen(expressionAST.as<ClassExpression>(), genContext);
        }
        else if (kind == SyntaxKind::OmittedExpression)
        {
            return mlirGen(expressionAST.as<OmittedExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ExpressionWithTypeArguments)
        {
            return mlirGen(expressionAST.as<ExpressionWithTypeArguments>(), genContext);
        }
        else if (kind == SyntaxKind::Unknown /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::success();
        }

        llvm_unreachable("unknown expression");
    }

    void inferType(mlir::Location location, mlir::Type templateType, mlir::Type concreteType, StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto currentTemplateType = templateType;
        auto currentType = concreteType;

        LLVM_DEBUG(llvm::dbgs() << "\n!! inferring \n\ttemplate type: " << templateType << ", \n\ttype: " << concreteType
                                << "\n";);

        if (!currentTemplateType || !currentType)
        {
            // nothing todo here
            return;
        }                                

        if (currentTemplateType == currentType)
        {
            // nothing todo here
            return;
        }

        if (auto namedGenType = dyn_cast<mlir_ts::NamedGenericType>(currentTemplateType))
        {
            // merge if exists

            auto name = namedGenType.getName().getValue();
            auto existType = results.lookup(name);
            if (existType)
            {
                auto merged = false;
                currentType = mth.mergeType(location, existType, currentType, merged);

                LLVM_DEBUG(llvm::dbgs() << "\n!! result type: " << currentType << "\n";);
                results[name] = currentType;
            }
            else
            {
                // TODO: when u use literal type to validate extends u need to use original type
                // currentType = mth.wideStorageType(currentType);
                LLVM_DEBUG(llvm::dbgs() << "\n!! type: " << name << " = " << currentType << "\n";);
                results.insert({name, currentType});
            }

            assert(results.lookup(name) == currentType);

            return;
        }

        // class -> class
        if (auto tempClass = dyn_cast<mlir_ts::ClassType>(currentTemplateType))
        {
            if (auto typeClass = dyn_cast<mlir_ts::ClassType>(concreteType))
            {
                auto typeClassInfo = getClassInfoByFullName(typeClass.getName().getValue());
                if (auto tempClassInfo = getClassInfoByFullName(tempClass.getName().getValue()))
                {
                    for (auto &templateParam : tempClassInfo->typeParamsWithArgs)
                    {
                        auto name = templateParam.getValue().first->getName();
                        auto found = typeClassInfo->typeParamsWithArgs.find(name);
                        if (found != typeClassInfo->typeParamsWithArgs.end())
                        {
                            // TODO: convert GenericType -> AnyGenericType,  and NamedGenericType -> GenericType, and
                            // add 2 type Parameters to it Constrain, Default
                            currentTemplateType = templateParam.getValue().second;
                            currentType = found->getValue().second;

                            inferType(location, currentTemplateType, currentType, results, genContext);
                        }
                    }

                    return;
                }
                else if (auto tempGenericClassInfo = getGenericClassInfoByFullName(tempClass.getName().getValue()))
                {
                    for (auto &templateParam : tempGenericClassInfo->typeParams)
                    {
                        auto name = templateParam->getName();
                        auto found = typeClassInfo->typeParamsWithArgs.find(name);
                        if (found != typeClassInfo->typeParamsWithArgs.end())
                        {
                            currentTemplateType = getNamedGenericType(found->getValue().first->getName());
                            currentType = found->getValue().second;

                            inferType(location, currentTemplateType, currentType, results, genContext);
                        }
                    }

                    return;
                }
            }
        }

        // interface -> interface
        if (auto tempInterface = dyn_cast<mlir_ts::InterfaceType>(currentTemplateType))
        {
            if (auto typeInterface = dyn_cast<mlir_ts::InterfaceType>(concreteType))
            {
                auto typeInterfaceInfo = getInterfaceInfoByFullName(typeInterface.getName().getValue());
                if (auto tempInterfaceInfo = getInterfaceInfoByFullName(tempInterface.getName().getValue()))
                {
                    for (auto &templateParam : tempInterfaceInfo->typeParamsWithArgs)
                    {
                        auto name = templateParam.getValue().first->getName();
                        auto found = typeInterfaceInfo->typeParamsWithArgs.find(name);
                        if (found != typeInterfaceInfo->typeParamsWithArgs.end())
                        {
                            // TODO: convert GenericType -> AnyGenericType,  and NamedGenericType -> GenericType, and
                            // add 2 type Parameters to it Constrain, Default
                            currentTemplateType = templateParam.getValue().second;
                            currentType = found->getValue().second;

                            inferType(location, currentTemplateType, currentType, results, genContext);
                        }
                    }

                    return;
                }
                else if (auto tempGenericInterfaceInfo = getGenericInterfaceInfoByFullName(tempInterface.getName().getValue()))
                {
                    for (auto &templateParam : tempGenericInterfaceInfo->typeParams)
                    {
                        auto name = templateParam->getName();
                        auto found = typeInterfaceInfo->typeParamsWithArgs.find(name);
                        if (found != typeInterfaceInfo->typeParamsWithArgs.end())
                        {
                            currentTemplateType = getNamedGenericType(found->getValue().first->getName());
                            currentType = found->getValue().second;

                            inferType(location, currentTemplateType, currentType, results, genContext);
                        }
                    }

                    return;
                }
            }
        }

        // array -> array
        if (auto tempArray = dyn_cast<mlir_ts::ArrayType>(currentTemplateType))
        {
            if (auto typeArray = dyn_cast<mlir_ts::ArrayType>(concreteType))
            {
                currentTemplateType = tempArray.getElementType();
                currentType = typeArray.getElementType();
                inferType(location, currentTemplateType, currentType, results, genContext);
                return;
            }

            if (auto typeArray = dyn_cast<mlir_ts::ConstArrayType>(concreteType))
            {
                currentTemplateType = tempArray.getElementType();
                currentType = typeArray.getElementType();
                inferType(location, currentTemplateType, currentType, results, genContext);
                return;
            }
        }

        // TODO: finish it
        // tuple -> tuple
        if (auto tempTuple = dyn_cast<mlir_ts::TupleType>(currentTemplateType))
        {
            if (auto typeTuple = dyn_cast<mlir_ts::TupleType>(concreteType))
            {
                for (auto tempFieldInfo : tempTuple.getFields())
                {
                    currentTemplateType = tempFieldInfo.type;
                    auto index = typeTuple.getIndex(tempFieldInfo.id);
                    if (index >= 0)
                    {
                        currentType = typeTuple.getFieldInfo(index).type;
                        inferType(location, currentTemplateType, currentType, results, genContext);
                    }
                    else
                    {
                        return;
                    }
                }

                return;
            }

            if (auto typeTuple = dyn_cast<mlir_ts::ConstTupleType>(concreteType))
            {
                for (auto tempFieldInfo : tempTuple.getFields())
                {
                    currentTemplateType = tempFieldInfo.type;
                    auto index = typeTuple.getIndex(tempFieldInfo.id);
                    if (index >= 0)
                    {
                        currentType = typeTuple.getFieldInfo(index).type;
                        inferType(location, currentTemplateType, currentType, results, genContext);
                    }
                    else
                    {
                        return;
                    }
                }

                return;
            }
        }        

        // optional -> optional
        if (auto tempOpt = dyn_cast<mlir_ts::OptionalType>(currentTemplateType))
        {
            if (auto typeOpt = dyn_cast<mlir_ts::OptionalType>(concreteType))
            {
                currentTemplateType = tempOpt.getElementType();
                currentType = typeOpt.getElementType();
                inferType(location, currentTemplateType, currentType, results, genContext);
                return;
            }

            // optional -> value
            currentTemplateType = tempOpt.getElementType();
            currentType = concreteType;
            inferType(location, currentTemplateType, currentType, results, genContext);
            return;
        }

        // lambda -> lambda
        if (mth.isAnyFunctionType(currentTemplateType) && mth.isAnyFunctionType(concreteType))
        {
            auto tempfuncType = mth.getParamsFromFuncRef(currentTemplateType);
            if (tempfuncType.size() > 0)
            {
                auto funcType = mth.getParamsFromFuncRef(concreteType);
                if (funcType.size() > 0)
                {
                    inferTypeFuncType(location, tempfuncType, funcType, results, genContext);

                    // lambda(return) -> lambda(return)
                    auto tempfuncRetType = mth.getReturnsFromFuncRef(currentTemplateType);
                    if (tempfuncRetType.size() > 0)
                    {
                        auto funcRetType = mth.getReturnsFromFuncRef(concreteType);
                        if (funcRetType.size() > 0)
                        {
                            inferTypeFuncType(location, tempfuncRetType, funcRetType, results, genContext);
                        }
                    }

                    return;
                }
            }
        }

        // union -> union
        if (auto tempUnionType = dyn_cast<mlir_ts::UnionType>(currentTemplateType))
        {
            if (auto typeUnionType = dyn_cast<mlir_ts::UnionType>(concreteType))
            {
                auto types = typeUnionType.getTypes();
                if (types.size() != tempUnionType.getTypes().size())
                {
                    return;
                }

                for (auto [index, tempSubType] : enumerate(tempUnionType.getTypes()))
                {
                    auto typeSubType = types[index];

                    currentTemplateType = tempSubType;
                    currentType = typeSubType;
                    inferType(location, currentTemplateType, currentType, results, genContext);
                }

                return;
            }
            else 
            {
                // TODO: review how to call functions such as: "function* Map<T, R>(a: T[] | Iterable<T>, f: (i: T) => R) { ... }"
                // special case when UnionType is used in generic method
                for (auto tempSubType : tempUnionType.getTypes())
                {
                    currentTemplateType = tempSubType;
                    currentType = concreteType;

                    auto count = results.size();
                    inferType(location, currentTemplateType, currentType, results, genContext);
                    if (count < results.size())
                    {
                        return;
                    }
                }

                return;
            }
        }

        // conditional type
        if (auto templateCondType = dyn_cast<mlir_ts::ConditionalType>(currentTemplateType))
        {
            currentTemplateType = templateCondType.getTrueType();
            inferType(location, currentTemplateType, currentType, results, genContext);
            currentTemplateType = templateCondType.getFalseType();
            inferType(location, currentTemplateType, currentType, results, genContext);
        }

        // typeref -> type
        if (auto tempTypeRefType = dyn_cast<mlir_ts::TypeReferenceType>(currentTemplateType))
        {
            currentTemplateType = getTypeByTypeReference(location, tempTypeRefType, genContext);
            inferType(location, currentTemplateType, currentType, results, genContext);
        }
    }

    void inferTypeFuncType(mlir::Location location, mlir::ArrayRef<mlir::Type> tempfuncType, mlir::ArrayRef<mlir::Type> funcType,
                           StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        if (tempfuncType.size() != funcType.size())
        {
            return;
        }

        for (auto paramIndex = 0; paramIndex < tempfuncType.size(); paramIndex++)
        {
            auto currentTemplateType = tempfuncType[paramIndex];
            auto currentType = funcType[paramIndex];
            inferType(location, currentTemplateType, currentType, results, genContext);
        }
    }

    bool isGenericFunctionReference(mlir::Value functionRefValue)
    {
        auto currValue = functionRefValue;
        if (auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return symbolOp->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME);
        }

        return false;
    }

    mlir::Type instantiateSpecializedFunctionTypeHelper(mlir::Location location, mlir::Value functionRefValue,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext)
    {
        auto currValue = functionRefValue;
        if (auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            auto functionName = symbolOp.getIdentifier();

            // it is not generic arrow function
            auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(functionName);

            MLIRNamespaceGuard nsGuard(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;

            return instantiateSpecializedFunctionTypeHelper(location, functionGenericTypeInfo->functionDeclaration,
                                                            recieverType, discoverReturnType, genContext);
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type instantiateSpecializedFunctionTypeHelper(mlir::Location location, FunctionLikeDeclarationBase funcDecl,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext)
    {
        GenContext funcGenContext(genContext);
        funcGenContext.receiverFuncType = recieverType;

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(theModule.getBody());

        auto [result, funcOp] = getFuncArgTypesOfGenericMethod(funcDecl, {}, discoverReturnType, funcGenContext);
        if (mlir::failed(result))
        {
            if (!genContext.dummyRun)
            {
                emitError(location) << "can't instantiate specialized arrow function.";
            }

            return mlir::Type();
        }

        return funcOp->getFuncType();
    }

    ValueOrLogicalResult instantiateSpecializedFunction(mlir::Location location,
        mlir::Value functionRefValue, mlir::Type recieverType, const GenContext &genContext)
    {
        auto currValue = functionRefValue;
        auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>();
        if (createBoundFunctionOp)
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>();
        assert(symbolOp);
        auto functionName = symbolOp.getIdentifier();

        // it is not generic arrow function
        auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(functionName);
        if (!functionGenericTypeInfo)
        {
            emitError(location) << "can't find information about generic function. " << functionName;
            return mlir::failure();            
        }

        GenContext funcGenContext(genContext);
        funcGenContext.receiverFuncType = recieverType;
        funcGenContext.specialization = true;
        funcGenContext.instantiateSpecializedFunction = true;
        funcGenContext.typeParamsWithArgs = functionGenericTypeInfo->typeParamsWithArgs;

        if (mlir::failed(processTypeArgumentsFromFunctionParameters(
            functionGenericTypeInfo->functionDeclaration, funcGenContext)))
        {
            emitError(location) << "can't instantiate specialized function from function parameters.";
            return mlir::failure();
        }

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            MLIRNamespaceGuard nsGuard(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;

            auto [result, specFuncOp, specFuncName, isGeneric] =
                mlirGenFunctionLikeDeclaration(functionGenericTypeInfo->functionDeclaration, funcGenContext);
            if (mlir::failed(result))
            {
                emitError(location) << "can't instantiate specialized function.";
                return mlir::failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! fixing spec. func: " << specFuncName << " type: ["
                                    << specFuncOp.getFunctionType() << "\n";);

            // update symbolref
            currValue.setType(specFuncOp.getFunctionType());
            if (functionName != specFuncName)
            {
                symbolOp.setIdentifier(specFuncName);
            }

            if (createBoundFunctionOp)
            {
                auto funcType = specFuncOp.getFunctionType();
                // fix create bound if any
                mlir::TypeSwitch<mlir::Type>(createBoundFunctionOp.getType())
                    .template Case<mlir_ts::BoundFunctionType>([&](auto boundFunc) {
                        functionRefValue.setType(getBoundFunctionType(funcType));
                    })
                    .template Case<mlir_ts::HybridFunctionType>([&](auto hybridFuncType) {
                        functionRefValue.setType(
                            mlir_ts::HybridFunctionType::get(builder.getContext(), funcType));
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }

            symbolOp->removeAttr(GENERIC_ATTR_NAME);

            builder.setInsertionPoint(symbolOp);

            // TODO: append captures vars to generic arrow function
            auto newOpWithCapture = resolveFunctionWithCapture(
                location, StringRef(specFuncName), specFuncOp.getFunctionType(), mlir::Value(), false, genContext);
            if (!newOpWithCapture.getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                // symbolOp will be removed as unsed
                return newOpWithCapture;
            }
            else
            {
                // newOpWithCapture will be removed as unsed
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult appendInferredTypes(mlir::Location location,
                                            llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                            StringMap<mlir::Type> &inferredTypes, IsGeneric &anyNamedGenericType,
                                            GenContext &genericTypeGenContext,
                                            bool arrayMerge = false, bool noExtendsTest = false)
    {
        for (auto &pair : inferredTypes)
        {
            // find typeParam
            auto typeParamName = pair.getKey();
            auto inferredType = pair.getValue();
            auto found = std::find_if(typeParams.begin(), typeParams.end(),
                                      [&](auto &paramItem) { return paramItem->getName() == typeParamName; });
            if (found == typeParams.end())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! can't find : " << typeParamName << " in type params: " << "\n";);
                LLVM_DEBUG(for (auto typeParam : typeParams) llvm::dbgs() << "\t!! type param: " << typeParam->getName() << "\n";);

                // experiment
                //auto typeParameterDOM = std::make_shared<TypeParameterDOM>(typeParamName.str());
                //genericTypeGenContext.typeParamsWithArgs[typeParamName] = {typeParameterDOM, inferredType};
                
                //return mlir::failure();
                // just ignore it
                continue;
            }

            auto typeParam = (*found);

            // we need to find out type and constrains is not allowing to do it
            auto [result, hasAnyNamedGenericType] =
                zipTypeParameterWithArgument(location, genericTypeGenContext.typeParamsWithArgs, typeParam,
                                             inferredType, noExtendsTest, genericTypeGenContext, true, arrayMerge);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (hasAnyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasAnyNamedGenericType;
            }
        }

        return mlir::success();
    }

    std::pair<mlir::LogicalResult, bool> resolveGenericParamFromFunctionCall(mlir::Location location, mlir::Type paramType, mlir::Value argOp, int paramIndex,
        GenericFunctionInfo::TypePtr functionGenericTypeInfo, IsGeneric &anyNamedGenericType,  GenContext &genericTypeGenContext)
    {
        if (paramType == argOp.getType())
        {
            return {mlir::success(), true};
        }

        StringMap<mlir::Type> inferredTypes;
        inferType(location, paramType, argOp.getType(), inferredTypes, genericTypeGenContext);
        if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                genericTypeGenContext, false, true)))
        {
            return {mlir::failure(), true};
        }

        if (isGenericFunctionReference(argOp))
        {
            GenContext typeGenContext(genericTypeGenContext);
            typeGenContext.dummyRun = true;
            auto recreatedFuncType = instantiateSpecializedFunctionTypeHelper(
                location, functionGenericTypeInfo->functionDeclaration, mlir::Type(), false,
                typeGenContext);
            if (!recreatedFuncType)
            {
                // next param
                return {mlir::failure(), true};
            }

            LLVM_DEBUG(llvm::dbgs()
                            << "\n!! instantiate specialized  type function: '"
                            << functionGenericTypeInfo->name << "' type: " << recreatedFuncType << "\n";);

            auto recreatedParamType = mth.getParamFromFuncRef(recreatedFuncType, paramIndex);

            LLVM_DEBUG(llvm::dbgs()
                            << "\n!! param type for arrow func[" << paramIndex << "]: " << recreatedParamType << "\n";);

            auto newArrowFuncType = instantiateSpecializedFunctionTypeHelper(location, argOp, recreatedParamType,
                                                                                true, genericTypeGenContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized arrow type function: "
                                    << newArrowFuncType << "\n";);

            if (!newArrowFuncType)
            {
                return {mlir::failure(), true};
            }

            // infer second type when ArrowType is fully built
            StringMap<mlir::Type> inferredTypes;
            inferType(location, paramType, newArrowFuncType, inferredTypes, genericTypeGenContext);
            if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                    genericTypeGenContext, false, true)))
            {
                return {mlir::failure(), false};
            }
        }

        return {mlir::success(), true};
    }

    mlir::LogicalResult resolveGenericParamsFromFunctionCall(mlir::Location location,
                                                             GenericFunctionInfo::TypePtr functionGenericTypeInfo,
                                                             NodeArray<TypeNode> typeArguments,
                                                             bool skipThisParam,
                                                             IsGeneric &anyNamedGenericType,
                                                             GenContext &genericTypeGenContext)
    {
        // add provided type arguments, ignoring defaults
        auto typeParams = functionGenericTypeInfo->typeParams;
        if (typeArguments)
        {
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArgumentsNoDefaults(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (hasAnyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasAnyNamedGenericType;
            }
        }

        // TODO: investigate, in [...].reduce, lambda function does not have funcOp, why?
        auto funcOp = functionGenericTypeInfo->funcOp;
        assert(funcOp);
        if (funcOp)
        {
            // TODO: we have func params.
            for (auto paramInfo : funcOp->getParams())
            {
                paramInfo->processed = false;
            }

            auto callOpsCount = genericTypeGenContext.callOperands.size();
            auto totalProcessed = 0;
            do
            {
                auto paramIndex = -1;
                auto processed = 0;
                auto startParamIndex = skipThisParam ? 1 : 0;
                auto skipCount = startParamIndex;
                for (auto paramInfo : funcOp->getParams())
                {
                    if (skipCount-- > 0)
                    {
                        continue;
                    }

                    paramIndex++;
                    if (paramInfo->processed)
                    {
                        continue;
                    }

                    auto paramType = paramInfo->getType();

                    if (callOpsCount <= paramIndex)
                    {
                        // there is no more ops
                        if (paramInfo->getIsOptional() || isa<mlir_ts::OptionalType>(paramType))
                        {
                            processed++;
                            continue;
                        }

                        if (paramInfo->getIsMultiArgsParam())
                        {
                            processed++;
                            continue;
                        }

                        break;
                    }

                    auto argOp = genericTypeGenContext.callOperands[paramIndex];

                    LLVM_DEBUG(llvm::dbgs()
                        << "\n!! resolving param for generic function: '"
                        << functionGenericTypeInfo->name << "'\n\t parameter #" << paramIndex << " type: [ " << paramType << " ] \n\t argument type: [ " << argOp.getType() << " ]\n";);

                    if (!paramInfo->getIsMultiArgsParam())
                    {
                        auto [result, cont] = resolveGenericParamFromFunctionCall(
                            location, paramType, argOp, paramIndex + startParamIndex, functionGenericTypeInfo, anyNamedGenericType, genericTypeGenContext);
                        if (mlir::succeeded(result))
                        {
                            paramInfo->processed = true;
                            processed++;
                        }
                        else if (!cont)
                        {
                            return mlir::failure();
                        }
                    }
                    else
                    {
                        struct ArrayInfo arrayInfo{};
                        for (auto varArgIndex = paramIndex; varArgIndex < callOpsCount; varArgIndex++)
                        {
                            auto argOp = genericTypeGenContext.callOperands[varArgIndex];

                            accumulateArrayItemType(location, argOp.getType(), arrayInfo);                            
                        }

                        mlir::Type arrayType = getArrayType(arrayInfo.accumulatedArrayElementType);

                        StringMap<mlir::Type> inferredTypes;
                        inferType(location, paramType, arrayType, inferredTypes, genericTypeGenContext);
                        if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                                genericTypeGenContext, true)))
                        {
                            return mlir::failure();
                        }                        

                        paramInfo->processed = true;
                        processed++;
                    }
                }

                if (processed == 0)
                {
                    emitError(location) << "not all types could be inferred";
                    return mlir::failure();
                }

                totalProcessed += processed;

                if (totalProcessed == funcOp->getParams().size() - startParamIndex)
                {
                    break;
                }

                if (totalProcessed > funcOp->getParams().size() + 100)
                {
                    // TODO: find out the issue
                    emitError(location) << "loop detected.";
                    return mlir::failure();
                }
            } while (true);
        }

        // add default params if not provided
        auto [resultDefArg, hasNamedGenericType] = zipTypeParametersWithDefaultArguments(
            location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);
        if (mlir::failed(resultDefArg))
        {
            return mlir::failure();
        }

        if (hasNamedGenericType == IsGeneric::True)
        {
            anyNamedGenericType = hasNamedGenericType;
        }

        // TODO: check if all typeParams are there
        if (genericTypeGenContext.typeParamsWithArgs.size() < typeParams.size())
        {
            // no resolve needed, this type without param
            emitError(location) << "not all types could be inferred";
            return mlir::failure();
        }

        return mlir::success();
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FunctionType, std::string> instantiateSpecializedFunction(
        mlir::Location location, StringRef name, NodeArray<TypeNode> typeArguments, bool skipThisParam, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(name);
        if (functionGenericTypeInfo)
        {
            if (functionGenericTypeInfo->functionDeclaration == SyntaxKind::ArrowFunction 
                || functionGenericTypeInfo->functionDeclaration == SyntaxKind::FunctionExpression)
            {
                // we need to avoid wrong redeclaration of arrow functions (when thisType is provided it will add THIS parameter as first)
                const_cast<GenContext &>(genContext).thisType = nullptr;
            }

            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;

            auto anyNamedGenericType = IsGeneric::False;

            // step 1, add type arguments first
            GenContext genericTypeGenContext(genContext);
            genericTypeGenContext.specialization = true;
            genericTypeGenContext.instantiateSpecializedFunction = true;
            genericTypeGenContext.typeParamsWithArgs = functionGenericTypeInfo->typeParamsWithArgs;
            auto typeParams = functionGenericTypeInfo->typeParams;
            if (typeArguments && typeParams.size() == typeArguments.size())
            {
                // create typeParamsWithArgs from typeArguments
                auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                    location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                if (hasAnyNamedGenericType == IsGeneric::True)
                {
                    anyNamedGenericType = hasAnyNamedGenericType;
                }
            }
            else if (genericTypeGenContext.callOperands.size() > 0 ||
                     functionGenericTypeInfo->functionDeclaration->parameters.size() > 0)
            {
                auto result =
                    resolveGenericParamsFromFunctionCall(location, functionGenericTypeInfo, typeArguments,
                                                         skipThisParam, anyNamedGenericType, genericTypeGenContext);
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            // we need to wide all types when initializing function
            // TODO: add checking constraints
            for (auto &typeParam : genericTypeGenContext.typeParamsWithArgs)
            {
                auto &typeParamValue = typeParam.getValue();
                auto typeInfo = std::get<0>(typeParamValue);
                auto name = typeInfo->getName();
                auto type = std::get<1>(typeParamValue);
                auto widenType = mth.wideStorageType(type);
                genericTypeGenContext.typeParamsWithArgs[name] = std::make_pair(typeInfo, widenType);

                if (typeParam.getValue().first->getConstraint())
                {
                    auto reason = testConstraint(location, genericTypeGenContext.typeParamsWithArgs, typeParamValue.first, widenType, genContext);
                    if (reason == Reason::Failure)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed. should be resolved later\n";);
                        return {mlir::failure(), mlir_ts::FunctionType(), ""};
                    }

                    if (reason == Reason::FailedConstraint)
                    {
                        if (functionGenericTypeInfo->funcType.getNumResults() > 0
                            && mlir::isa<mlir_ts::TypePredicateType>(functionGenericTypeInfo->funcType.getResult(0)))
                        {
                            return {
                                mlir::success(), 
                                mlir_ts::FunctionType::get(builder.getContext(), {}, { getBooleanLiteral(false) }, false), 
                                ""
                            };
                        }

                        return {mlir::failure(), mlir_ts::FunctionType(), ""};
                    }
                }
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized function: " << functionGenericTypeInfo->name
                                    << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // revalidate all types
            if (anyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = IsGeneric::False;
                for (auto &typeParamWithArg : genericTypeGenContext.typeParamsWithArgs)
                {
                    if (mth.isGenericType(std::get<1>(typeParamWithArg.second)))
                    {
                        anyNamedGenericType = IsGeneric::True;
                    }
                }
            }

            if (anyNamedGenericType == IsGeneric::False)
            {
                if (functionGenericTypeInfo->processing)
                {
                    auto [fullName, name] =
                        getNameOfFunction(functionGenericTypeInfo->functionDeclaration, genericTypeGenContext);

                    auto funcType = lookupFunctionTypeMap(fullName);
                    if (funcType)
                    {
                        return {mlir::success(), funcType, fullName};
                    }

                    if (genContext.allowPartialResolve)
                    {
                        return {mlir::success(), mlir_ts::FunctionType(), fullName};
                    }

                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                // create new instance of function with TypeArguments
                functionGenericTypeInfo->processing = true;
                auto [result, funcOp, funcName, isGeneric] =
                    mlirGenFunctionLikeDeclaration(functionGenericTypeInfo->functionDeclaration, genericTypeGenContext);
                functionGenericTypeInfo->processing = false;
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                functionGenericTypeInfo->processed = true;

                // instatiate all ArrowFunctions which are not yet instantiated
                auto opIndex = skipThisParam ? 0 : -1;
                for (auto [callOpIndex, op] : enumerate(genContext.callOperands))
                {
                    opIndex++;
                    if (isGenericFunctionReference(op))
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! delayed arrow func instantiation for func type: "
                                                << funcOp.getFunctionType() << "\n";);
                        auto result = instantiateSpecializedFunction(
                            location, op, funcOp.getFunctionType().getInput(opIndex), genContext);
                        if (mlir::failed(result))
                        {
                            return {mlir::failure(), mlir_ts::FunctionType(), ""};
                        }

                        auto resultValue = V(result);
                        if (resultValue)
                        {
                            operands[callOpIndex] = resultValue;
                        }
                    }
                }

                return {mlir::success(), funcOp.getFunctionType(), funcOp.getName().str()};
            }

            emitError(location) << "can't instantiate specialized function [" << name << "].";
            return {mlir::failure(), mlir_ts::FunctionType(), ""};
        }

        emitError(location) << "can't find generic [" << name << "] function.";
        return {mlir::failure(), mlir_ts::FunctionType(), ""};
    }

    std::pair<mlir::LogicalResult, FunctionPrototypeDOM::TypePtr> getFuncArgTypesOfGenericMethod(
        FunctionLikeDeclarationBase functionLikeDeclarationAST, ArrayRef<TypeParameterDOM::TypePtr> typeParams,
        bool discoverReturnType, const GenContext &genContext)
    {
        GenContext funcGenContext(genContext);
        funcGenContext.discoverParamsOnly = !discoverReturnType;

        // we need to map generic parameters to generic types to be able to resolve function parameters which
        // are not generic
        for (auto typeParam : typeParams)
        {
            funcGenContext.typeAliasMap.insert({typeParam->getName(), getNamedGenericType(typeParam->getName())});
        }

        auto [funcOp, funcProto, result, isGenericType] =
            mlirGenFunctionPrototype(functionLikeDeclarationAST, funcGenContext);
        if (mlir::failed(result) || !funcOp)
        {
            return {mlir::failure(), {}};
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! func name: " << funcProto->getName()
                                << ", Op type (resolving from operands): " << funcOp.getFunctionType() << "\n";);

        LLVM_DEBUG(llvm::dbgs() << "\n!! func args: "; for (auto [index, paramInfo]
                                                                            : enumerate(funcProto->getParams())) {
            llvm::dbgs() << "\n_ " << paramInfo->getName() << ": " << paramInfo->getType() << " = (" << index << ") ";
            if (genContext.callOperands.size() > index)
                llvm::dbgs() << genContext.callOperands[index];
            llvm::dbgs() << "\n";
        });

        return {mlir::success(), funcProto};
    }

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               NodeArray<TypeNode> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics = false)
    {
        auto fullNameGenericClassTypeName = genericClassType.getName().getValue();
        auto genericClassInfo = getGenericClassInfoByFullName(fullNameGenericClassTypeName);
        if (genericClassInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericClassInfo->elementNamespace;

            GenContext genericTypeGenContext(genContext);
            genericTypeGenContext.instantiateSpecializedFunction = false;
            auto typeParams = genericClassInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized class: " << fullNameGenericClassTypeName << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // create new instance of interface with TypeArguments
            if (mlir::failed(std::get<0>(mlirGen(genericClassInfo->classDeclaration, genericTypeGenContext))))
            {
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationClassType(genericClassInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // special case: Array<T>
        if (fullNameGenericClassTypeName == "Array" && typeArguments.size() == 1)
        {
            auto arraySpecType = getEmbeddedTypeWithParam(fullNameGenericClassTypeName, typeArguments, genContext);
            return {mlir::success(), arraySpecType};
        }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               ArrayRef<mlir::Type> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics = false)
    {
        auto fullNameGenericClassTypeName = genericClassType.getName().getValue();
        auto genericClassInfo = getGenericClassInfoByFullName(fullNameGenericClassTypeName);
        if (genericClassInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericClassInfo->elementNamespace;

            GenContext genericTypeGenContext(genContext);
            genericTypeGenContext.instantiateSpecializedFunction = false;
            auto typeParams = genericClassInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized class: " << fullNameGenericClassTypeName << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            static auto count = 0;
            count++;
            if (count > 99)
            {
                count--;
                emitError(location) << "can't instantiate type. '" << genericClassType
                                    << "'. Circular initialization is detected.";
                return {mlir::failure(), mlir::Type()};

                // std::string s;
                // s += "can't instantiate type. '";
                // s += fullNameGenericClassTypeName;
                // s += "'. Circular initialization is detected.";
                // llvm_unreachable(s.c_str());
            }

            auto res = std::get<0>(mlirGen(genericClassInfo->classDeclaration, genericTypeGenContext));
            count--;

            // create new instance of class with TypeArguments
            if (mlir::failed(res))
            {
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationClassType(genericClassInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedInterfaceType(
        mlir::Location location, mlir_ts::InterfaceType genericInterfaceType, NodeArray<TypeNode> typeArguments,
        const GenContext &genContext, bool allowNamedGenerics = false)
    {
        auto fullNameGenericInterfaceTypeName = genericInterfaceType.getName().getValue();
        auto genericInterfaceInfo = getGenericInterfaceInfoByFullName(fullNameGenericInterfaceTypeName);
        if (genericInterfaceInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericInterfaceInfo->elementNamespace;

            GenContext genericTypeGenContext(genContext);
            auto typeParams = genericInterfaceInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized interface: " << fullNameGenericInterfaceTypeName
                                    << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // create new instance of interface with TypeArguments
            if (mlir::failed(mlirGen(genericInterfaceInfo->interfaceDeclaration, genericTypeGenContext)))
            {
                // return mlir::Type();
                // type can't be resolved, so return generic base type
                //return {mlir::success(), genericInterfaceInfo->interfaceType};
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationInterfaceType(genericInterfaceInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    ValueOrLogicalResult mlirGenSpecialized(mlir::Location location, mlir::Value genResult,
                                            NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        // in case it is generic arrow function
        auto currValue = genResult;

        // in case of this.generic_func<T>();
        if (auto extensFuncRef = currValue.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
        {
            currValue = extensFuncRef.getFunc();

            SmallVector<mlir::Value, 4> operands;
            operands.push_back(extensFuncRef.getThisVal());
            operands.append(genContext.callOperands.begin(), genContext.callOperands.end());

            GenContext specGenContext(genContext);
            specGenContext.callOperands = operands;

            auto newFuncRefOrLogicResult = mlirGenSpecialized(location, currValue, typeArguments, operands, specGenContext);
            EXIT_IF_FAILED(newFuncRefOrLogicResult)
            if (newFuncRefOrLogicResult && currValue != newFuncRefOrLogicResult)
            {
                mlir::Value newFuncRefValue = newFuncRefOrLogicResult;

                // special case to work with interfaces
                // TODO: finish it, bug
                auto thisRef = extensFuncRef.getThisVal();
                auto funcType = mlir::cast<mlir_ts::FunctionType>(newFuncRefValue.getType());

                mlir::Value newExtensionFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                                location, getExtensionFunctionType(funcType), thisRef, newFuncRefValue);

                extensFuncRef.erase();

                return newExtensionFuncVal;
            }
            else
            {
                return genResult;
            }
        }

        if (currValue.getDefiningOp()->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME))
        {
            // create new function instance
            GenContext initSpecGenContext(genContext);
            initSpecGenContext.forceDiscover = true;
            initSpecGenContext.thisType = mlir::Type();

            auto skipThisParam = false;
            mlir::Value thisValue;
            StringRef funcName;
            if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                funcName = symbolOp.getIdentifierAttr().getValue();
            }
            else if (auto thisSymbolOp = currValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                funcName = thisSymbolOp.getIdentifierAttr().getValue();
                skipThisParam = true;
                thisValue = thisSymbolOp.getThisVal();
                initSpecGenContext.thisType = thisValue.getType();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            auto [result, funcType, funcSymbolName] =
                instantiateSpecializedFunction(location, funcName, typeArguments, skipThisParam, operands, initSpecGenContext);
            if (mlir::failed(result))
            {
                emitError(location) << "can't instantiate function. '" << funcName
                                    << "' not all generic types can be identified";
                return mlir::failure();
            }

            if (!funcType && genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            return resolveFunctionWithCapture(location, StringRef(funcSymbolName), funcType, thisValue, false, genContext);
        }

        if (auto classOp = genResult.getDefiningOp<mlir_ts::ClassRefOp>())
        {
            auto classType = classOp.getType();
            auto [result, specType] = instantiateSpecializedClassType(location, classType, typeArguments, genContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (auto specClassType = dyn_cast_or_null<mlir_ts::ClassType>(specType))
            {
                return V(builder.create<mlir_ts::ClassRefOp>(
                    location, specClassType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), specClassType.getName().getValue())));
            }

            if (specType)
            {
                return V(builder.create<mlir_ts::TypeRefOp>(location, specType));
            }

            return genResult;
        }

        if (auto ifaceOp = genResult.getDefiningOp<mlir_ts::InterfaceRefOp>())
        {
            auto interfaceType = ifaceOp.getType();
            auto [result, specType] =
                instantiateSpecializedInterfaceType(location, interfaceType, typeArguments, genContext);
            if (auto specInterfaceType = dyn_cast_or_null<mlir_ts::InterfaceType>(specType))
            {
                return V(builder.create<mlir_ts::InterfaceRefOp>(
                    location, specInterfaceType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), specInterfaceType.getName().getValue())));
            }

            return genResult;
        }

        return genResult;
    }

    ValueOrLogicalResult mlirGen(Expression expression, NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto genResult = V(result);
        if (typeArguments.size() == 0)
        {
            return genResult;
        }

        auto location = loc(expression);

        SmallVector<mlir::Value, 4> emptyOperands;
        return mlirGenSpecialized(location, genResult, typeArguments, emptyOperands, genContext);
    }

    ValueOrLogicalResult mlirGen(ExpressionWithTypeArguments expressionWithTypeArgumentsAST,
                                 const GenContext &genContext)
    {
        return mlirGen(expressionWithTypeArgumentsAST->expression, expressionWithTypeArgumentsAST->typeArguments,
                       genContext);
    }

    ValueOrLogicalResult registerVariableInThisContext(mlir::Location location, StringRef name, mlir::Type type,
                                                       const GenContext &genContext)
    {
        if (genContext.passResult)
        {

            // create new type with added field
            genContext.passResult->extraFieldsInThisContext.push_back({MLIRHelper::TupleFieldName(name, builder.getContext()), type});
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

        MLIRCodeLogic mcl(builder);
        auto thisVarValueRef = mcl.GetReferenceFromValue(location, thisVarValue);

        assert(thisVarValueRef);

        return V(thisVarValueRef);
    }

    bool isConstValue(mlir::Value init)
    {
        if (!init)
        {
            return false;
        }

        if (isa<mlir_ts::ConstArrayType>(init.getType()) || isa<mlir_ts::ConstTupleType>(init.getType()))
        {
            return true;
        }

        auto defOp = init.getDefiningOp();
        if (isa<mlir_ts::ConstantOp>(defOp) || isa<mlir_ts::UndefOp>(defOp) || isa<mlir_ts::NullOp>(defOp))
        {
            return true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! is it const? : " << init << "\n";);

        return false;
    }

    struct VariableDeclarationInfo
    {
        VariableDeclarationInfo() : variableName(), fullName(), initial(), type(), storage(), globalOp(), varClass(),
            scope{VariableScope::Local}, isFullName{false}, isGlobal{false}, isConst{false}, isExternal{false}, isExport{false}, isImport{false}, 
            isSpecialization{false}, allocateOutsideOfOperation{false}, allocateInContextThis{false}, comdat{Select::NotSet}, deleted{false}, isUsed{false}
        {
        };

        VariableDeclarationInfo(
            TypeValueInitFuncType func_, 
            std::function<StringRef(StringRef)> getFullNamespaceName_) : VariableDeclarationInfo() 
        {
            getFullNamespaceName = getFullNamespaceName_;
            func = func_;
        }

        void setName(StringRef name_)
        {
            variableName = name_;
            fullName = name_;

            // I think it is only making it worst
            if (!isFullName && isGlobal)
                fullName = getFullNamespaceName(name_);
        }        

        void setType(mlir::Type type_)
        {
            type = type_;
        }             

        void setInitial(mlir::Value initial_)
        {
            initial = initial_;
        }             

        void setIsTypeProvided(TypeProvided typeProvided_)
        {
            typeProvided = typeProvided_;
        }

        void setExternal(bool isExternal_)
        {
            isExternal = isExternal_;
        }        

        void setStorage(mlir::Value storage_)
        {
            storage = storage_;
        }   

        void setSpecialization()
        {
            isSpecialization = true;
        }   

        void detectFlags(bool isFullName_, VariableClass varClass_, const GenContext &genContext)
        {
            varClass = varClass_;
            isFullName = isFullName_;
            
            if (isFullName_ || !genContext.funcOp)
            {
                scope = VariableScope::Global;
            }

            allocateOutsideOfOperation = genContext.allocateVarsOutsideOfOperation
                || genContext.allocateUsingVarsOutsideOfOperation && varClass_.isUsing;
            allocateInContextThis = genContext.allocateVarsInContextThis;

            isGlobal = scope == VariableScope::Global || varClass == VariableType::Var;
            isConst = (varClass == VariableType::Const || varClass == VariableType::ConstRef) &&
                       !allocateOutsideOfOperation && !allocateInContextThis;
            isExternal = varClass == VariableType::External;
            isExport = varClass.isExport;
            isImport = varClass.isImport;
            isPublic = varClass.isPublic;
            isAppendingLinkage = varClass.isAppendingLinkage;
            comdat = varClass.comdat;
            isUsed = varClass.isUsed;
        }

        mlir::LogicalResult processConstRef(mlir::Location location, mlir::OpBuilder &builder, const GenContext &genContext)
        {
            if (mlir::failed(getVariableTypeAndInit(location, genContext)))
            {
                return mlir::failure();
            }

            if (varClass == VariableType::ConstRef)
            {
                MLIRCodeLogic mcl(builder);
                if (auto possibleInit = mcl.GetReferenceFromValue(location, initial))
                {
                    setInitial(possibleInit);
                }
                else
                {
                    // convert ConstRef to Const again as this is const object (it seems)
                    varClass = VariableType::Const;
                }
            }

            return mlir::success();
        }

        mlir::LogicalResult getVariableTypeAndInit(mlir::Location location, const GenContext &genContext)
        {
            auto [type, init, typeProvided] = func(location, genContext);
            if (!type)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "Can't resolve variable '" << variableName << "' type";
                }

                return mlir::failure();
            }

            if (isa<mlir_ts::VoidType>(type))
            {
                emitError(location) << "variable '" << variableName << "' can't be 'void' type";
                return mlir::failure();
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                emitError(location) << "variable '" << variableName << "' can't be 'never' type";
                return mlir::failure();
            }

            assert(type);
            setType(type);
            setInitial(init);
            setIsTypeProvided(typeProvided);

            return mlir::success();
        }    

        VariableDeclarationDOM::TypePtr createVariableDeclaration(mlir::Location location, const GenContext &genContext)
        {
            auto varDecl = std::make_shared<VariableDeclarationDOM>(fullName, type, location);
            if (!isConst || varClass == VariableType::ConstRef)
            {
                varDecl->setReadWriteAccess();
                // TODO: HACK: to mark var as local and ignore when capturing
                if (varClass == VariableType::ConstRef)
                {
                    varDecl->setIgnoreCapturing();
                }
            }

            varDecl->setUsing(varClass.isUsing);

            return varDecl;
        }

        bool getIsPublic()
        {
            return isExternal || isExport || isPublic;
        }

        LLVM::Linkage getLinkage()
        {
            auto linkage = LLVM::Linkage::Private;
            if (isExternal || comdat != Select::NotSet)
            {
                linkage = LLVM::Linkage::External;
            }
            else if (isAppendingLinkage)
            {
                linkage = LLVM::Linkage::Appending;
            }
            else if (isSpecialization)
            {
                linkage = LLVM::Linkage::LinkonceODR;
                // TODO: dso_local somehow linked with -fno-pic
                //attrs.push_back({builder.getStringAttr("dso_local"), builder.getUnitAttr()});
            }
            else if (isExport || isImport || isPublic)
            {
                linkage = LLVM::Linkage::External;
            }

            return linkage;            
        }

        void printDebugInfo()
        {
            LLVM_DEBUG(dbgs() << "\n!! variable = " << fullName << " type: " << type << "\n";);
        }

        TypeValueInitFuncType func;
        std::function<StringRef(StringRef)> getFullNamespaceName;

        StringRef variableName;
        StringRef fullName;
        mlir::Value initial;
        TypeProvided typeProvided;
        mlir::Type type;
        mlir::Value storage;
        mlir_ts::GlobalOp globalOp;

        VariableClass varClass;
        VariableScope scope;
        bool isFullName;
        bool isGlobal;
        bool isConst;
        bool isExternal;
        bool isPublic;
        bool isExport;
        bool isImport;
        bool isAppendingLinkage;
        bool isSpecialization;
        bool allocateOutsideOfOperation;
        bool allocateInContextThis;
        Select comdat;
        bool deleted;
        bool isUsed;
    };

    mlir::LogicalResult adjustLocalVariableType(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        auto type = variableDeclarationInfo.type;

        // if it is Optional type, we need to set to undefined                
        if (isa<mlir_ts::OptionalType>(type) && !variableDeclarationInfo.initial)
        {                    
            CAST_A(castedValue, location, type, getUndefined(location), genContext);
            variableDeclarationInfo.setInitial(castedValue);
        }

        if (variableDeclarationInfo.isConst)
        {
            return mlir::success();
        }

        auto actualType = variableDeclarationInfo.typeProvided == TypeProvided::Yes ? type : mth.wideStorageType(type);

        // this is 'let', if 'let' is func, it should be HybridFunction
        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(actualType))
        {
            actualType = mlir_ts::HybridFunctionType::get(builder.getContext(), funcType);
        }

        if (variableDeclarationInfo.initial && actualType != type)
        {
            CAST_A(castedValue, location, actualType, variableDeclarationInfo.initial, genContext);
            variableDeclarationInfo.setInitial(castedValue);
        }

        variableDeclarationInfo.setType(actualType);

        return mlir::success();
    }

    mlir::LogicalResult adjustGlobalVariableType(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        if (variableDeclarationInfo.isConst)
        {
            return mlir::success();
        }

        auto type = variableDeclarationInfo.type;

        auto actualType = variableDeclarationInfo.typeProvided == TypeProvided::Yes ? type : mth.wideStorageType(type);

        variableDeclarationInfo.setType(actualType);

        if (variableDeclarationInfo.initial && actualType != type)
        {
            // cast
            auto result = cast(location, actualType, variableDeclarationInfo.initial, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            variableDeclarationInfo.initial = V(result);
        }

        return mlir::success();
    }    
   
    mlir::LogicalResult createLocalVariable(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        if (mlir::failed(variableDeclarationInfo.getVariableTypeAndInit(location, genContext)))
        {
            return mlir::failure();
        }

        if (mlir::failed(adjustLocalVariableType(location, variableDeclarationInfo, genContext)))
        {
            return mlir::failure();
        }

        if (mth.isGenericType(variableDeclarationInfo.type))
        {
            genContext.postponedMessages->clear();
            emitError(location) << "variable '" 
                << variableDeclarationInfo.variableName << "' is referencing generic type." 
                << (mth.isAnyFunctionType(variableDeclarationInfo.type) ? " use 'const' instead of 'let'" : "");
            return mlir::failure();
        }

        // scope to restore inserting point
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            if (variableDeclarationInfo.allocateOutsideOfOperation)
            {
                builder.setInsertionPoint(genContext.currentOperation);
            }

            if (variableDeclarationInfo.allocateInContextThis)
            {
                auto varValueInThisContext = registerVariableInThisContext(location, variableDeclarationInfo.variableName, variableDeclarationInfo.type, genContext);
                variableDeclarationInfo.setStorage(varValueInThisContext);
            }

            if (!variableDeclarationInfo.storage)
            {
                // default case
                auto varOpValue = builder.create<mlir_ts::VariableOp>(
                    location, mlir_ts::RefType::get(variableDeclarationInfo.type),
                    variableDeclarationInfo.allocateOutsideOfOperation ? mlir::Value() : variableDeclarationInfo.initial,
                    builder.getBoolAttr(false), builder.getIndexAttr(0));

                variableDeclarationInfo.setStorage(varOpValue);
            }
        }

        // init must be in its normal place
        if ((variableDeclarationInfo.allocateInContextThis || variableDeclarationInfo.allocateOutsideOfOperation) 
            && variableDeclarationInfo.initial 
            && variableDeclarationInfo.storage)
        {
            builder.create<mlir_ts::StoreOp>(location, variableDeclarationInfo.initial, variableDeclarationInfo.storage);
        }

        return mlir::success();
    }    

    mlir::LogicalResult createGlobalVariableInitialization(mlir::Location location, mlir_ts::GlobalOp globalOp, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard insertGuard(builder);

        auto &region = globalOp.getInitializerRegion();
        auto *block = builder.createBlock(&region);

        builder.setInsertionPoint(block, block->begin());

        GenContext genContextWithNameReceiver(genContext);
        if (variableDeclarationInfo.isConst)
        {
            genContextWithNameReceiver.receiverName = variableDeclarationInfo.fullName;
        }
        else
        {
            genContextWithNameReceiver.isGlobalVarReceiver = true;
        }

        if (mlir::failed(variableDeclarationInfo.getVariableTypeAndInit(location, genContextWithNameReceiver)))
        {
            return mlir::failure();
        }

        if (mlir::failed(adjustGlobalVariableType(location, variableDeclarationInfo, genContext)))
        {
            return mlir::failure();
        }        

        globalOp.setTypeAttr(mlir::TypeAttr::get(variableDeclarationInfo.type));
        /*
        if (variableDeclarationInfo.isExport)
        {
            addGlobalToExport(variableDeclarationInfo.variableName, variableDeclarationInfo.type, genContext);
        }
        */

        if (!variableDeclarationInfo.initial)
        {
            //variableDeclarationInfo.initial = builder.create<mlir_ts::UndefOp>(location, variableDeclarationInfo.type);
            variableDeclarationInfo.initial = builder.create<mlir_ts::DefaultOp>(location, variableDeclarationInfo.type);
        }

        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{variableDeclarationInfo.initial});

        return mlir::success();
    }    

    mlir::LogicalResult createGlobalVariableUndefinedInitialization(mlir::Location location, mlir_ts::GlobalOp globalOp, struct VariableDeclarationInfo &variableDeclarationInfo)
    {
        // we need to put undefined into GlobalOp
        mlir::OpBuilder::InsertionGuard insertGuard(builder);

        auto &region = globalOp.getInitializerRegion();
        auto *block = builder.createBlock(&region);

        builder.setInsertionPoint(block, block->begin());

        auto undefVal = builder.create<mlir_ts::UndefOp>(location, variableDeclarationInfo.type);
        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{undefVal});

        return mlir::success();
    }

    mlir::LogicalResult createGlobalVariable(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        // generate only for real pass
        mlir_ts::GlobalOp globalOp;
        // get constant
        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(theModule.getBody());
            // find last string
            auto lastUse = [&](mlir::Operation *op) {
                if (auto globalOp = dyn_cast<mlir_ts::GlobalOp>(op))
                {
                    builder.setInsertionPointAfter(globalOp);
                }
            };

            theModule.getBody()->walk(lastUse);

            SmallVector<mlir::NamedAttribute> attrs;

            // add modifiers
            if (variableDeclarationInfo.isExport)
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), "export"), mlir::UnitAttr::get(builder.getContext())});
            }            

            if (variableDeclarationInfo.isImport)
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), "import"), mlir::UnitAttr::get(builder.getContext())});
            }  

            if (this->compileOptions.generateDebugInfo)
            {
                MLIRDebugInfoHelper mti(builder, debugScope);
                auto namedLoc = mti.combineWithCurrentScopeAndName(location, variableDeclarationInfo.variableName);
                location = namedLoc;
            }

            globalOp = builder.create<mlir_ts::GlobalOp>(
                location, builder.getNoneType(), variableDeclarationInfo.isConst, variableDeclarationInfo.fullName, variableDeclarationInfo.getLinkage(), attrs);                

            if (variableDeclarationInfo.comdat != Select::NotSet)
            {
                globalOp.setComdatAttr(builder.getI32IntegerAttr(static_cast<int32_t>(variableDeclarationInfo.comdat)));
            }  

            variableDeclarationInfo.globalOp = globalOp;

            if (genContext.dummyRun && genContext.cleanUpOps)
            {
                genContext.cleanUpOps->push_back(globalOp);
            }

            if (variableDeclarationInfo.scope == VariableScope::Global)
            {
                if (variableDeclarationInfo.getIsPublic())
                {
                    globalOp.setPublic();
                }
                else 
                {
                    globalOp.setPrivate();
                }

                if (variableDeclarationInfo.isExternal)
                {
                    if (mlir::failed(variableDeclarationInfo.getVariableTypeAndInit(location, genContext)))
                    {
                        return mlir::failure();
                    }

                    if (mlir::failed(adjustGlobalVariableType(location, variableDeclarationInfo, genContext)))
                    {
                        return mlir::failure();
                    }                      

                    globalOp.setTypeAttr(mlir::TypeAttr::get(variableDeclarationInfo.type));
                }
                else
                {
                    createGlobalVariableInitialization(location, globalOp, variableDeclarationInfo, genContext);
                }

                if (variableDeclarationInfo.isUsed)
                {
                    builder.setInsertionPointAfter(globalOp);
                    builder.create<mlir_ts::AppendToUsedOp>(location, globalOp.getName());
                }

                return mlir::success();
            }
        }

        // it is not global scope (for example 'var' in function)
        if (mlir::failed(variableDeclarationInfo.getVariableTypeAndInit(location, genContext))) 
        {
            return mlir::failure();
        }

        if (mlir::failed(adjustGlobalVariableType(location, variableDeclarationInfo, genContext)))
        {
            return mlir::failure();
        }  

        globalOp.setTypeAttr(mlir::TypeAttr::get(variableDeclarationInfo.type));
        if (variableDeclarationInfo.isExternal)
        {
            // all is done here
            return mlir::success();
        }

        if (variableDeclarationInfo.initial)
        {
            // save value
            auto address = builder.create<mlir_ts::AddressOfOp>(
                location, mlir_ts::RefType::get(variableDeclarationInfo.type), variableDeclarationInfo.fullName, mlir::IntegerAttr());
            builder.create<mlir_ts::StoreOp>(location, variableDeclarationInfo.initial, address);
        }

        auto result = createGlobalVariableUndefinedInitialization(location, globalOp, variableDeclarationInfo);

        if (variableDeclarationInfo.isUsed)
        {
            builder.setInsertionPointAfter(globalOp);
            builder.create<mlir_ts::AppendToUsedOp>(location, globalOp.getName());
        }

        return result;
    }    

    mlir::LogicalResult isGlobalConstLambda(mlir::Location location, struct VariableDeclarationInfo &variableDeclarationInfo, const GenContext &genContext)
    {
        if (variableDeclarationInfo.isConst 
            && variableDeclarationInfo.initial 
            && mth.isAnyFunctionType(variableDeclarationInfo.type))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult registerVariableDeclaration(mlir::Location location, VariableDeclarationDOM::TypePtr variableDeclaration, struct VariableDeclarationInfo &variableDeclarationInfo, bool showWarnings, const GenContext &genContext)
    {
        if (variableDeclarationInfo.deleted)
        {
            return mlir::success();
        }
        else if (!variableDeclarationInfo.isGlobal)
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
                && !mth.isGenericType(variableDeclarationInfo.initial.getType())
                && !mth.isAnyFunctionType(variableDeclarationInfo.initial.getType()))
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
        }
        else
        {
            getGlobalsMap().insert({variableDeclarationInfo.variableName, variableDeclaration});
        }

        return mlir::success();
    }

    mlir::Type registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                                TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false)
    {
        struct VariableDeclarationInfo variableDeclarationInfo(
            func, std::bind(&MLIRGenImpl::getGlobalsFullNamespaceName, this, std::placeholders::_1));

        variableDeclarationInfo.detectFlags(isFullName, varClass, genContext);
        variableDeclarationInfo.setName(name);

        if (declarationMode)
            variableDeclarationInfo.setExternal(true);

        if (!variableDeclarationInfo.isGlobal)
        {
            if (variableDeclarationInfo.isConst)
                variableDeclarationInfo.processConstRef(location, builder, genContext);
            else if (mlir::failed(createLocalVariable(location, variableDeclarationInfo, genContext)))
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
                variableDeclarationInfo.globalOp->erase();
                variableDeclarationInfo.deleted = true;
            }
        }

        if (!variableDeclarationInfo.type)
        {
            emitError(location) << "type of variable '" << variableDeclarationInfo.variableName << "' is not valid";
            return variableDeclarationInfo.type;
        }

        //LLVM_DEBUG(variableDeclarationInfo.printDebugInfo(););

        auto varDecl = variableDeclarationInfo.createVariableDeclaration(location, genContext);
        if (genContext.usingVars != nullptr && varDecl->getUsing())
        {
            genContext.usingVars->push_back(varDecl);
        }

        registerVariableDeclaration(location, varDecl, variableDeclarationInfo, showWarnings, genContext);
        return varDecl->getType();
    }

    // TODO: to support '...' u need to use 'processOperandSpreadElement' and instead of "index" param use "next" logic
    ValueOrLogicalResult processDeclarationArrayBindingPatternSubPath(mlir::Location location, int index, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, init, builder.getI32IntegerAttr(index));
        mlir::Value subInit =
            mlir::TypeSwitch<mlir::Type, mlir::Value>(type)
                .template Case<mlir_ts::ConstTupleType>(
                    [&](auto constTupleType) { return cl.Tuple(constTupleType, true); })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) { return cl.Tuple(tupleType, true); })
                .template Case<mlir_ts::ConstArrayType>([&](auto constArrayType) {
                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                        builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                        location, mlir_ts::RefType::get(constArrayType.getElementType()), init, constIndex);
                    return builder.create<mlir_ts::LoadOp>(location, constArrayType.getElementType(), elemRef);
                })
                .template Case<mlir_ts::ArrayType>([&](auto arrayType) {
                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                        builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                        location, mlir_ts::RefType::get(arrayType.getElementType()), init, constIndex);
                    return builder.create<mlir_ts::LoadOp>(location, arrayType.getElementType(), elemRef);
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); return mlir::Value(); });

        if (!subInit)
        {
            return mlir::failure();
        }

        return subInit; 
    }

    mlir::LogicalResult processDeclarationArrayBindingPattern(mlir::Location location, ArrayBindingPattern arrayBindingPattern,
                                               VariableClass varClass,
                                               TypeValueInitFuncType func,
                                               const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

        for (auto [index, arrayBindingElement] : enumerate(arrayBindingPattern->elements))
        {
            auto subValueFunc = [&](mlir::Location location, const GenContext &genContext) { 
                auto result = processDeclarationArrayBindingPatternSubPath(location, index, type, init, genContext);
                if (result.failed_or_no_value()) 
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No); 
                }

                auto value = V(result);
                return std::make_tuple(value.getType(), value, TypeProvided::No); 
            };

            if (arrayBindingElement == SyntaxKind::BindingElement && mlir::failed(processDeclaration(
                    arrayBindingElement.as<BindingElement>(), varClass, subValueFunc, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Attribute getFieldNameFromBindingElement(BindingElement objectBindingElement)
    {
        mlir::Attribute fieldName;
        if (objectBindingElement->propertyName == SyntaxKind::NumericLiteral)
        {
            fieldName = getNumericLiteralAttribute(objectBindingElement->propertyName);
        }
        else
        {
            auto propertyName = MLIRHelper::getName(objectBindingElement->propertyName);
            if (propertyName.empty())
            {
                propertyName = MLIRHelper::getName(objectBindingElement->name);
            }

            if (!propertyName.empty())
            {
                fieldName = MLIRHelper::TupleFieldName(propertyName, builder.getContext());
            }
        }

        return fieldName;
    }

    ValueOrLogicalResult processDeclarationObjectBindingPatternSubPath(
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
            MLIRPropertyAccessCodeLogic cl(builder, location, init, fieldName);
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                value = cl.Tuple(tupleType, true);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                value = cl.Tuple(constTupleType, true);
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

        if (objectBindingElement->initializer)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            auto subType = mlir::cast<mlir_ts::OptionalType>(tupleType.getFieldInfo(tupleType.getIndex(fieldName)).type).getElementType();
            auto res = optionalValueOrDefault(location, subType, value, objectBindingElement->initializer, genContext);
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

    ValueOrLogicalResult processDeclarationObjectBindingPatternSubPathSpread(
        mlir::Location location, ObjectBindingPattern objectBindingPattern, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        mlir::Value subInit;
        mlir::Type subInitType;

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

    mlir::LogicalResult processDeclarationObjectBindingPattern(mlir::Location location, ObjectBindingPattern objectBindingPattern,
                                                VariableClass varClass,
                                                TypeValueInitFuncType func,
                                                const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

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

    mlir::LogicalResult processDeclarationName(DeclarationName name, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false)
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
                addVariableDeclarationToExport(nameStr, currentNamespace, varType, isConst);
            }

            return mlir::success();
        }

        return mlir::failure();       
    }

    mlir::LogicalResult processDeclaration(NamedDeclaration item, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false)
    {
        if (item == SyntaxKind::OmittedExpression)
        {
            return mlir::success();
        }

        item->name->parent = item;
        return processDeclarationName(item->name, varClass, func, genContext, showWarnings);
    }

    template <typename ItemTy>
    TypeValueInitType getTypeOnly(ItemTy item, mlir::Type defaultType, const GenContext &genContext)
    {
        // type
        auto typeProvided = TypeProvided::No;
        mlir::Type type = defaultType;
        if (item->type)
        {
            type = getType(item->type, genContext);
            typeProvided = TypeProvided::Yes;
        }

        return std::make_tuple(type, mlir::Value(), typeProvided);
    }

    template <typename ItemTy>
    std::tuple<mlir::Type, bool, bool> evaluateTypeAndInit(ItemTy item, const GenContext &genContext)
    {
        // type
        auto hasInit = false;
        auto typeProvided = false;
        mlir::Type type;
        if (item->type)
        {
            type = getType(item->type, genContext);
            typeProvided = true;
        }

        // init
        if (auto initializer = item->initializer)
        {
            hasInit = true;
            auto initType = evaluate(initializer, genContext);
            if (initType && !type)
            {
                type = initType;
            }
        }

        return std::make_tuple(type, hasInit, typeProvided);
    }

    template <typename ItemTy>
    std::tuple<mlir::Type, mlir::Value, TypeProvided> getTypeAndInit(ItemTy item, const GenContext &genContext)
    {
        // type
        auto typeProvided = TypeProvided::No;
        mlir::Type type;
        if (item->type)
        {
            type = getType(item->type, genContext);
            if (!type || VALIDATE_FUNC_BOOL(type))
            {
                return {mlir::Type(), mlir::Value(), TypeProvided::No};
            }

            typeProvided = TypeProvided::Yes;
        }

        // init
        mlir::Value init;
        if (auto initializer = item->initializer)
        {
            GenContext genContextWithTypeReceiver(genContext);
            genContextWithTypeReceiver.clearReceiverTypes();
            if (type)
            {
                genContextWithTypeReceiver.receiverType = type;
                LLVM_DEBUG(dbgs() << "\n!! variable receiverType " << type << "\n");
            }

            auto result = mlirGen(initializer, genContextWithTypeReceiver);
            if (result.failed())
            {
                return {mlir::Type(), mlir::Value(), TypeProvided::No};
            }

            init = V(result);
            if (init)
            {
                if (!type)
                {
                    type = init.getType();
                }
                else if (type != init.getType())
                {
                    auto result = cast(loc(initializer), type, init, genContext);
                    if (result.failed())
                    {
                        return {mlir::Type(), mlir::Value(), TypeProvided::No};
                    }

                    init = V(result);
                }
            }
        }
        else if (typeProvided == TypeProvided::Yes && type && item == SyntaxKind::VariableDeclaration)
        {
            auto parent = item->parent;
            if (parent && parent == SyntaxKind::VariableDeclarationList)
            {
                parent = parent->parent;
                if (parent && parent == SyntaxKind::VariableStatement)
                {
                    // there is no initializer, var declration can be undefined
                    //type = getUnionType(type, getUndefinedType());
                    if (!isa<mlir_ts::OptionalType>(type) && !hasModifier(parent, SyntaxKind::DeclareKeyword))
                    {
                        emitWarning(loc(item), "'let' does not have initializer, use undefined union type '<type> | undefined'.");
                    }
                }
            }
        }

#ifdef ANY_AS_DEFAULT
        if (!type)
        {
            type = getAnyType();
        }
#endif

        return std::make_tuple(type, init, typeProvided);
    }

    mlir::LogicalResult mlirGen(VariableDeclaration item, VariableClass varClass, const GenContext &genContext)
    {
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

        auto initFunc = [&](mlir::Location location, const GenContext &genContext) {
            if (declarationMode)
            {
                auto [t, b, p] = evaluateTypeAndInit(item, genContext);
                return std::make_tuple(t, mlir::Value(), p ? TypeProvided::Yes : TypeProvided::No);
            }

            auto typeAndInit = getTypeAndInit(item, genContext);

            if (varClass.isDynamicImport)
            {
                auto nameStr = getFullNamespaceName(MLIRHelper::getName(item->name));
                auto fieldType = std::get<0>(typeAndInit);
                if (fieldType)
                {
                    auto dllVarName = V(mlirGenStringValue(location, nameStr, true));
                    auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                        location, getOpaqueType(), dllVarName);
                    auto refToTyped = cast(location, mlir_ts::RefType::get(fieldType), referenceToStaticFieldOpaque, genContext);
                    auto valueOfField = builder.create<mlir_ts::LoadOp>(location, fieldType, refToTyped);
                    return std::make_tuple(valueOfField.getType(), V(valueOfField), TypeProvided::Yes);                
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

                auto parentModule = theModule;
                MLIRCodeLogicHelper mclh(builder, location);

                builder.setInsertionPointToStart(parentModule.getBody());
                mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());                    

                builder.create<mlir_ts::GlobalConstructorOp>(
                    location, mlir::FlatSymbolRefAttr::get(builder.getContext(), fullInitGlobalFuncName), builder.getIndexAttr(LAST_GLOBAL_CONSTRUCTOR_PRIORITY));
            }
        }
        else if (mlir::failed(processDeclaration(item, valClassItem, initFunc, genContext, true)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    auto getExportModifier(Node node) -> boolean
    {
        if (compileOptions.exportOpt == ExportAll)
        {
            return true;
        }

        if (compileOptions.exportOpt == IgnoreAll)
        {
            return false;
        }

        return hasModifier(node, SyntaxKind::ExportKeyword);
    }

    mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
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

        if (variableDeclarationListAST->parent)
        {
            varClass.isPublic = hasModifier(variableDeclarationListAST->parent, SyntaxKind::ExportKeyword);
            varClass.isExport = getExportModifier(variableDeclarationListAST->parent);
            MLIRHelper::iterateDecorators(variableDeclarationListAST->parent, [&](std::string name, SmallVector<std::string> args) {
                if (name == DLL_EXPORT)
                {
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

                if (name == "used") {
                    varClass.isUsed = true;
                }
            });
        }

        for (auto &item : variableDeclarationListAST->declarations)
        {
            // we need it for support "undefined type" in 'let' without initialization
            item->parent = variableDeclarationListAST;
            if (mlir::failed(mlirGen(item, varClass, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
    {
        // we need it for support "export" keyword
        variableStatementAST->declarationList->parent = variableStatementAST;
        return mlirGen(variableStatementAST->declarationList, genContext);
    }

    mlir::LogicalResult mlirGenParameterBindingElement(BindingElement objectBindingElement, SmallVector<mlir_ts::FieldInfo> &fieldInfos, const GenContext &genContext)
    {
        auto fieldId = getFieldNameFromBindingElement(objectBindingElement);
        if (!fieldId)
        {
            auto genName = MLIRHelper::getAnonymousName(loc_check(objectBindingElement), ".be", "");
            fieldId = MLIRHelper::TupleFieldName(genName, builder.getContext());
        }

        mlir::Type fieldType;

        if (objectBindingElement->initializer)
        {
            auto evalType = evaluate(objectBindingElement->initializer, genContext);
            auto widenType = mth.wideStorageType(evalType);

            // if it has initializer - it should have optional type to support default values
            fieldType = getOptionalType(widenType);
        }
        else if (objectBindingElement->name == SyntaxKind::ObjectBindingPattern || objectBindingElement->name == SyntaxKind::ArrayBindingPattern)
        {
            fieldType = mlirGenParameterObjectOrArrayBinding(objectBindingElement->name, genContext);
        }
        else
        {
            emitError(loc(objectBindingElement)) << "can't resolve type for binding pattern '"
                                                << fieldId << "', provide default initializer";
            return mlir::failure();
        }

        LLVM_DEBUG(dbgs() << "\n!! property " << fieldId << " mapped to type " << fieldType << "");

        fieldInfos.push_back({fieldId, fieldType});

        return mlir::success();
    }

    mlir::Type mlirGenParameterObjectOrArrayBinding(Node name, const GenContext &genContext)
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
            auto arrayBindingPattern = name.as<ArrayBindingPattern>();
            for (auto arrayBindingElement : arrayBindingPattern->elements)
            {
                if (arrayBindingElement == SyntaxKind::OmittedExpression)
                {
                    continue;
                }

                if (arrayBindingElement == SyntaxKind::BindingElement)
                {
                    auto objectBindingElement = arrayBindingElement.as<BindingElement>();
                    mlirGenParameterBindingElement(objectBindingElement, fieldInfos, genContext);
                }
            }

            return getTupleType(fieldInfos);
        }        

        return mlir::Type();
    }

    bool isGenericParameters(SignatureDeclarationBase parametersContextAST, const GenContext &genContext)
    {
        if (parametersContextAST == SyntaxKind::GetAccessor || parametersContextAST == SyntaxKind::SetAccessor)
        {
            return false;
        }

        auto formalParams = parametersContextAST->parameters;
        for (auto [index, arg] : enumerate(formalParams))
        {
            auto isBindingPattern = arg->name == SyntaxKind::ObjectBindingPattern 
                || arg->name == SyntaxKind::ArrayBindingPattern;

            mlir::Type type;
            auto typeParameter = arg->type;

            auto location = loc(typeParameter);

            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                continue;
            }

            if (mth.isNoneType(type) && genContext.receiverFuncType && mth.isAnyFunctionType(genContext.receiverFuncType))
            {
                type = mth.getParamFromFuncRef(genContext.receiverFuncType, index);
                if (!type)
                {
                    return false;
                }
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
            }

            if (mth.isNoneType(type))
            {
                if (!typeParameter && !initializer)
                {
                    return true;
                }
            }
        }

        return false;
    }

    mlir::StringRef getArgumentName(int index) {
        std::stringstream ss;
        ss << "arg" << index;
        return mlir::StringRef(ss.str()).copy(stringAllocator);        
    }

    mlir::StringRef getParameterGenericTypeName(std::string name) {
        mlir::StringRef typeParamNamePtr;
        std::stringstream ss;
        ss << "TGenParam_" << name;
        return mlir::StringRef(ss.str()).copy(stringAllocator);
    }

    std::tuple<mlir::LogicalResult, bool, std::vector<std::shared_ptr<FunctionParamDOM>>> mlirGenParameters(
        SignatureDeclarationBase parametersContextAST, const GenContext &genContext)
    {
        // to remove variables such as "this" from scope after using it in params context
        SymbolTableScopeT varScope(symbolTable);

        auto isGenericTypes = false;
        std::vector<std::shared_ptr<FunctionParamDOM>> params;

        SyntaxKind kind = parametersContextAST;
        // add this param
        auto isStatic = 
            hasModifier(parametersContextAST->parent, SyntaxKind::StaticKeyword) 
            || hasModifier(parametersContextAST, SyntaxKind::StaticKeyword);

        if (parametersContextAST->parent == SyntaxKind::InterfaceDeclaration)
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, getOpaqueType(), loc(parametersContextAST)));
        }
        else if (!isStatic &&
            (kind == SyntaxKind::MethodDeclaration || kind == SyntaxKind::Constructor ||
             kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor))
        {
            params.push_back(
                std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }
        else if (!isStatic && genContext.thisType && !!parametersContextAST->parent &&
            (kind == SyntaxKind::FunctionExpression ||
             kind == SyntaxKind::ArrowFunction))
        {            
            // TODO: this is very tricky code, if we rediscover function again and if by any chance thisType is not null, it will append thisType to lambda which very wrong code
            params.push_back(
                std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }

        auto formalParams = parametersContextAST->parameters;
        for (auto [index, arg] : enumerate(formalParams))
        {
            auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
            if (namePtr.empty())
            {
                namePtr = getArgumentName(index);
            }

            auto isBindingPattern = arg->name == SyntaxKind::ObjectBindingPattern || arg->name == SyntaxKind::ArrayBindingPattern;

            mlir::Type type;
            auto isMultiArgs = !!arg->dotDotDotToken;
            auto isOptional = !!arg->questionToken;
            auto typeParameter = arg->type;

            auto location = loc(typeParameter);

            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // special case, setup 'this' and type provided 
            if (namePtr == THIS_NAME && type) 
            {
                const_cast<GenContext &>(genContext).thisType = type;
                LLVM_DEBUG(dbgs() << "\n!! param " << THIS_NAME << " mapped to type " << type << "\n");

                auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_NAME, type, location);
                auto typeRefVal = builder.create<mlir_ts::TypeRefOp>(location, type);
                declare(location, varDecl, typeRefVal, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                auto evalType = evaluate(initializer, genContext);
                if (evalType)
                {
                    evalType = mth.wideStorageType(evalType);

                    // TODO: set type if not provided
                    isOptional = true;
                    if (mth.isNoneType(type))
                    {
                        type = evalType;
                    }
                }
            }

            if (mth.isNoneType(type) && genContext.receiverFuncType && mth.isAnyFunctionType(genContext.receiverFuncType))
            {
                type = mth.getParamFromFuncRef(genContext.receiverFuncType, index);
                if (!type)
                {
                    emitError(location) << "can't resolve type for parameter '" << namePtr << "', the receiver function has less parameters.";
                    return {mlir::failure(), isGenericTypes, params};                    
                }

                LLVM_DEBUG(dbgs() << "\n!! param " << namePtr << " mapped to type " << type << "\n");

                isGenericTypes |= mth.isGenericType(type);
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
                LLVM_DEBUG(dbgs() << "\n!! binding param " << namePtr << " is type " << type << "\n");
            }

            if (mth.isNoneType(type))
            {
                if (!typeParameter && !initializer)
                {
#ifndef ANY_AS_DEFAULT
                    if (!genContext.allowPartialResolve && !genContext.dummyRun)
                    {
                        auto funcName = MLIRHelper::getName(parametersContextAST->name);
                        emitError(loc(arg))
                            << "type of parameter '" << namePtr
                            << "' is not provided, parameter must have type or initializer, function: " << funcName;
                    }
                    return {mlir::failure(), isGenericTypes, params};
#else
                    emitWarning(loc(parametersContextAST)) << "type for parameter '" << namePtr << "' is any";
                    type = getAnyType();
#endif
                }
                else
                {
                    emitError(location) << "can't resolve type for parameter '" << namePtr << "'";
                    return {mlir::failure(), isGenericTypes, params};
                }
            }

            if (isa<mlir_ts::VoidType>(type))
            {
                emitError(location, "'Void' can't be used as parameter type");
                return {mlir::failure(), isGenericTypes, params};
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                emitError(location, "'Never' can't be used as parameter type");
                return {mlir::failure(), isGenericTypes, params};
            }

            if (isBindingPattern)
            {
                params.push_back(
                    std::make_shared<FunctionParamDOM>(
                        namePtr, type, loc(arg), isOptional, isMultiArgs, initializer, arg->name));
            }
            else
            {
                params.push_back(
                    std::make_shared<FunctionParamDOM>(
                        namePtr, type, loc(arg), isOptional, isMultiArgs, initializer));
            }
        }

        return {mlir::success(), isGenericTypes, params};
    }

    std::tuple<std::string, std::string> getNameOfFunction(SignatureDeclarationBase signatureDeclarationBaseAST,
                                                           const GenContext &genContext)
    {
        auto name = getNameWithArguments(signatureDeclarationBaseAST, genContext);
        std::string objectOwnerName;
        if (signatureDeclarationBaseAST->parent == SyntaxKind::ClassDeclaration ||
            signatureDeclarationBaseAST->parent == SyntaxKind::ClassExpression)
        {
            objectOwnerName =
                getNameWithArguments(signatureDeclarationBaseAST->parent.as<ClassDeclaration>(), genContext);
        }
        else if (signatureDeclarationBaseAST->parent == SyntaxKind::InterfaceDeclaration)
        {
            objectOwnerName =
                getNameWithArguments(signatureDeclarationBaseAST->parent.as<InterfaceDeclaration>(), genContext);
        }
        else if (signatureDeclarationBaseAST->parent == SyntaxKind::ObjectLiteralExpression)
        {
            objectOwnerName = mlir::cast<mlir_ts::ObjectStorageType>(
                mlir::cast<mlir_ts::ObjectType>(genContext.thisType).getStorageType()).getName().getValue();
        }
        else if (genContext.funcOp)
        {
            auto funcName = const_cast<GenContext &>(genContext).funcOp.getSymName().str();
            objectOwnerName = funcName;
        }

        if (signatureDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            if (!objectOwnerName.empty())
            {
                // class method name
                name = objectOwnerName + "." + name;
            }
            else
            {
                name = MLIRHelper::getAnonymousName(loc_check(signatureDeclarationBaseAST), ".md", "");
            }
        }
        // TODO: for new () interfaces
        else if (signatureDeclarationBaseAST == SyntaxKind::MethodSignature 
                || signatureDeclarationBaseAST == SyntaxKind::ConstructSignature)
        {
            // class method name
            name = objectOwnerName + "." + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::GetAccessor)
        {
            // class method name
            name = objectOwnerName + ".get_" + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::SetAccessor)
        {
            // class method name
            name = objectOwnerName + ".set_" + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::Constructor)
        {
            // class method name
            auto isStatic = 
                hasModifier(signatureDeclarationBaseAST->parent, SyntaxKind::StaticKeyword)
                || hasModifier(signatureDeclarationBaseAST, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                name = objectOwnerName + "." + STATIC_NAME + "_" + name;
            }
            else
            {
                name = objectOwnerName + "." + name;
            }
        }

        auto fullName = getFullNamespaceName(name).str();
        return std::make_tuple(fullName, name);
    }

    // TODO: review it, seems doing work which mlirGenFunctionPrototype will overwrite anyway
    std::tuple<FunctionPrototypeDOM::TypePtr, mlir_ts::FunctionType, SmallVector<mlir::Type>>
    mlirGenFunctionSignaturePrototype(SignatureDeclarationBase signatureDeclarationBaseAST, bool defaultVoid,
                                      const GenContext &genContext)
    {
        auto [fullName, name] = getNameOfFunction(signatureDeclarationBaseAST, genContext);

        registerNamespace(name, true);

        mlir_ts::FunctionType funcType;
        auto [result, isGenericType, params] = mlirGenParameters(signatureDeclarationBaseAST, genContext);

        exitNamespace();

        if (mlir::failed(result))
        {
            return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
        }

        SmallVector<mlir::Type> argTypes;
        auto isMultiArgs = false;

        // auto isAsync = hasModifier(signatureDeclarationBaseAST, SyntaxKind::AsyncKeyword);

        for (const auto &param : params)
        {
            auto paramType = param->getType();
            if (mth.isNoneType(paramType))
            {
                return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
            }

            if (param->getIsOptional() && !isa<mlir_ts::OptionalType>(paramType))
            {
                argTypes.push_back(getOptionalType(paramType));
            }
            else
            {
                argTypes.push_back(paramType);
            }

            isMultiArgs |= param->getIsMultiArgsParam();
        }

        auto funcProto = std::make_shared<FunctionPrototypeDOM>(fullName, params);

        funcProto->setNameWithoutNamespace(name);
        funcProto->setIsGeneric(isGenericType);

        // check if function already discovered
        auto funcIt = getFunctionMap().find(name);
        if (funcIt != getFunctionMap().end())
        {
            auto cachedFuncType = funcIt->second.getFunctionType();
            if (cachedFuncType.getNumResults() > 0)
            {
                auto returnType = cachedFuncType.getResult(0);
                funcProto->setReturnType(returnType);
            }

            funcType = cachedFuncType;
        }
        else if (auto typeParameter = signatureDeclarationBaseAST->type)
        {
            GenContext paramsGenContext(genContext);
            paramsGenContext.funcProto = funcProto;

            auto returnType = getType(typeParameter, paramsGenContext);
            if (!returnType)
            {
                return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
            }

            funcProto->setReturnType(returnType);

            funcType = getFunctionType(argTypes, returnType, isMultiArgs);
        }
        else if (defaultVoid)
        {
            auto returnType = getVoidType();
            funcProto->setReturnType(returnType);

            funcType = getFunctionType(argTypes, returnType, isMultiArgs);
        }

        return std::make_tuple(funcProto, funcType, argTypes);
    }

    bool isGlobalAttr(StringRef name)
    {
        static llvm::StringMap<bool> funcAttrs {
            {"optnone", true },
            {DLL_IMPORT, true },
            {DLL_EXPORT, true },
        };

        return funcAttrs[name];        
    }

    bool isFuncAttr(StringRef name)
    {
        static llvm::StringMap<bool> funcAttrs {
            {"noinline", true },
            {"optnone", true },
            {DLL_IMPORT, true },
            {DLL_EXPORT, true },
        };

        return funcAttrs[name];        
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, mlir::LogicalResult, bool> mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        mlir_ts::FuncOp funcOp;

        auto [funcProto, funcType, argTypes] =
            mlirGenFunctionSignaturePrototype(
                functionLikeDeclarationBaseAST, 
                hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::DeclareKeyword), 
                genContext);
        if (!funcProto)
        {
            return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
        }

        GenContext funcProtoGenContext(genContext);
        funcProtoGenContext.funcProto = funcProto;

        auto fullName = funcProto->getName();

        mlir_ts::FunctionType functionDiscovered;
        auto funcTypeIt = getFunctionTypeMap().find(fullName);
        if (funcTypeIt != getFunctionTypeMap().end())
        {
            functionDiscovered = (*funcTypeIt).second;
        }        

        // discover type & args
        // seems we need to discover it all the time due to captured vars
        auto detectReturnType = (!funcType || funcProtoGenContext.forceDiscover || !functionDiscovered)
            && !funcProto->getIsGeneric();
        if (detectReturnType)
        {
            if (mlir::succeeded(discoverFunctionReturnTypeAndCapturedVars(functionLikeDeclarationBaseAST, fullName,
                                                                          argTypes, funcProto, funcProtoGenContext)))
            {
                if (!funcProtoGenContext.forceDiscover && funcType && funcType.getNumResults() > 0)
                {
                    funcProto->setReturnType(funcType.getResult(0));
                }
                else if (auto typeParameter = functionLikeDeclarationBaseAST->type)
                {
                    // rewrite ret type with actual value in case of specialized generic
                    auto returnType = getType(typeParameter, funcProtoGenContext);
                    funcProto->setReturnType(returnType);
                }
                else if (funcProtoGenContext.receiverFuncType)
                {
                    // rewrite ret type with actual value
                    auto &argTypeDestFuncType = funcProtoGenContext.receiverFuncType;
                    auto retTypeFromReceiver = mth.isAnyFunctionType(argTypeDestFuncType) 
                        ? mth.getReturnTypeFromFuncRef(argTypeDestFuncType)
                        : mlir::Type();
                    if (retTypeFromReceiver 
                        && !mth.isNoneType(retTypeFromReceiver) 
                        && !mth.isGenericType(retTypeFromReceiver))
                    {
                        funcProto->setReturnType(retTypeFromReceiver);
                        LLVM_DEBUG(llvm::dbgs()
                                       << "\n!! set return type from receiver: " << retTypeFromReceiver << "\n";);
                    }
                }

                // create funcType
                if (funcProto->getReturnType())
                {
                    funcType = getFunctionType(argTypes, funcProto->getReturnType(), funcProto->isMultiArgs());
                }
                else
                {
                    // no return type
                    funcType = getFunctionType(argTypes, std::nullopt, funcProto->isMultiArgs());
                }
            }
            else
            {
                // false result
                return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
            }
        }
        else if (functionDiscovered)
        {
            funcType = functionDiscovered;
        }

        // we need it, when we run rediscovery second time
        if (!funcProto->getHasExtraFields())
        {
            funcProto->setHasExtraFields(existLocalVarsInThisContextMap(funcProto->getName()));
        }

        SmallVector<mlir::NamedAttribute> attrs;
#ifdef ADD_GC_ATTRIBUTE
        attrs.push_back({builder.getIdentifier(TS_GC_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
#endif
        // add decorations, "noinline, optnone"

        MLIRHelper::iterateDecorators(functionLikeDeclarationBaseAST, [&](std::string name, SmallVector<std::string> args) {
            if (isFuncAttr(name))
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), name), mlir::UnitAttr::get(builder.getContext())});
            }

            if (name == "varargs") 
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), "func.varargs"), mlir::BoolAttr::get(builder.getContext(), true)});
            }

            if (name == "used") {
                builder.create<mlir_ts::AppendToUsedOp>(location, funcProto->getName());
            }
        });

        // add modifiers
        auto dllExport = getExportModifier(functionLikeDeclarationBaseAST)
            || ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllExport) == InternalFlags::DllExport);
        if (dllExport)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "export"), mlir::UnitAttr::get(builder.getContext())});
        }

        auto dllImport = ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllImport) == InternalFlags::DllImport);
        if (dllImport)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "import"), mlir::UnitAttr::get(builder.getContext())});
        }

        if (funcProtoGenContext.specialization)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "specialization"), mlir::UnitAttr::get(builder.getContext())});
        }
            
        if (funcType)
        {
            auto it = getCaptureVarsMap().find(funcProto->getName());
            auto hasCapturedVars = funcProto->getHasCapturedVars() || (it != getCaptureVarsMap().end());
            if (hasCapturedVars)
            {
                // important set when it is discovered and in process second type
                funcProto->setHasCapturedVars(true);
                funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs);
            }
            else
            {
                funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs);
            }

            funcProto->setFuncType(funcType);

            if (dllExport)
            {
                if (functionLikeDeclarationBaseAST == SyntaxKind::FunctionDeclaration
                    || functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction)
                {
                    addFunctionDeclarationToExport(funcProto, currentNamespace);
                }
            }
        }

        if (!funcProto->getIsGeneric())
        {
            auto funcTypeIt = getFunctionTypeMap().find(fullName);
            if (funcTypeIt != getFunctionTypeMap().end())
            {
                getFunctionTypeMap().erase(funcTypeIt);
            }

            getFunctionTypeMap().insert({fullName, funcType});

            LLVM_DEBUG(llvm::dbgs() << "\n!! register func name: " << fullName << ", type: " << funcType << "\n";);
        }

        return std::make_tuple(funcOp, funcProto, mlir::success(), funcProto->getIsGeneric());
    }

    mlir::LogicalResult discoverFunctionReturnTypeAndCapturedVars(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, StringRef name, SmallVector<mlir::Type> &argTypes,
        const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext)
    {
        if (funcProto->getDiscovered())
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n\tdiscovering 'return type' & 'captured variables' for : " << name << "\n";);

        mlir::OpBuilder::InsertionGuard guard(builder);

        auto partialDeclFuncType = getFunctionType(argTypes, std::nullopt, false);
        auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

        {
            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> 
                fullNameGlobalsMapScope(fullNameGlobalsMap);

            GenContext genContextWithPassResult{};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.thisType = genContext.thisType;
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = new SmallVector<mlir::Block *>();
            genContextWithPassResult.cleanUpOps = new SmallVector<mlir::Operation *>();
            genContextWithPassResult.passResult = new PassResult();
            genContextWithPassResult.state = new int(1);
            genContextWithPassResult.allocateVarsInContextThis =
                (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::VarsInObjectContext) ==
                InternalFlags::VarsInObjectContext;
            genContextWithPassResult.discoverParamsOnly = genContext.discoverParamsOnly;
            genContextWithPassResult.typeAliasMap = genContext.typeAliasMap;
            genContextWithPassResult.typeParamsWithArgs = genContext.typeParamsWithArgs;
            genContextWithPassResult.postponedMessages = genContext.postponedMessages;

            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, name, dummyFuncOp, funcProto,
                                              genContextWithPassResult)))
            {
                exitNamespace();

                auto &passResult = genContextWithPassResult.passResult;
                if (passResult->functionReturnTypeShouldBeProvided 
                    && mth.isNoneType(passResult->functionReturnType))
                {
                    // has return value but type is not provided yet
                    genContextWithPassResult.clean();
                    emitError(loc(functionLikeDeclarationBaseAST)) << "'return' is not found in function or return type can't be resolved";
                    return mlir::failure();
                }

                funcProto->setDiscovered(true);
                auto discoveredType = passResult->functionReturnType;
                if (discoveredType && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?

                    funcProto->setReturnType(mth.convertConstArrayTypeToArrayType(discoveredType));
                    LLVM_DEBUG(llvm::dbgs()
                                   << "\n!! ret type: " << funcProto->getReturnType() << ", name: " << name << "\n";);
                }

                // if we have captured parameters, add first param to send lambda's type(class)
                if (passResult->outerVariables.size() > 0)
                {
                    MLIRCodeLogic mcl(builder);
                    auto isObjectType =
                        genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
                    if (!isObjectType)
                    {
                        argTypes.insert(argTypes.begin(), mcl.CaptureType(passResult->outerVariables));
                    }

                    getCaptureVarsMap().insert({name, passResult->outerVariables});
                    funcProto->setHasCapturedVars(true);

                    LLVM_DEBUG(llvm::dbgs() << "\n!! has captured vars, name: " << name << "\n";);

                    LLVM_DEBUG(for (auto& var : passResult->outerVariables)
                    {
                        llvm::dbgs() << "\n!! ...captured var - name: " << var.second->getName() << ", type: " << var.second->getType() << "\n";
                    });
                }

                if (passResult->extraFieldsInThisContext.size() > 0)
                {
                    getLocalVarsInThisContextMap().insert({name, passResult->extraFieldsInThisContext});

                    funcProto->setHasExtraFields(true);
                }

                genContextWithPassResult.clean();

                LLVM_DEBUG(llvm::dbgs() << "\n\tSUCCESS - discovering 'return type' & 'captured variables' for : " << name << "\n";);

                return mlir::success();
            }
            else
            {
                exitNamespace();

                genContextWithPassResult.clean();

                LLVM_DEBUG(llvm::dbgs() << "\n\tFAILED - discovering 'return type' & 'captured variables' for : " << name << "\n";);

                return mlir::failure();
            }
        }
    }

    mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
    {
        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        // declaring function which is nested and object should not have this context (unless it is part of object declaration)
        if (!functionDeclarationAST->parent && funcGenContext.thisType != nullptr)
        {
            funcGenContext.thisType = nullptr;
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        auto res = mlirGenFunctionLikeDeclaration(functionDeclarationAST, funcGenContext);
        return std::get<0>(res);
    }

    ValueOrLogicalResult mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
    {
        auto location = loc(functionExpressionAST);
        mlir_ts::FuncOp funcOp;
        std::string funcName;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            // provide name for it
            auto funcGenContext = GenContext(genContext);
            funcGenContext.clearScopeVars();
            funcGenContext.thisType = nullptr;

            auto [result, funcOpRet, funcNameRet, isGenericRet] =
                mlirGenFunctionLikeDeclaration(functionExpressionAST, funcGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            funcOp = funcOpRet;
            funcName = funcNameRet;
        }

        // if funcOp is null, means lambda is generic]
        if (!funcOp)
        {
            // return reference to generic method
            if (getGenericFunctionMap().count(funcName))
            {
                auto genericFunctionInfo = getGenericFunctionMap().lookup(funcName);
                // info: it will not take any capture now
                return resolveFunctionWithCapture(location, genericFunctionInfo->name, genericFunctionInfo->funcType,
                                                  mlir::Value(), true, genContext);
            }
            else
            {
                emitError(location) << "can't find generic function: " << funcName;
                return mlir::failure();
            }
        }

        return resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getFunctionType(), mlir::Value(), false, genContext);
    }

    ValueOrLogicalResult mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
    {
        auto location = loc(arrowFunctionAST);
        mlir_ts::FuncOp funcOp;
        std::string funcName;
        bool isGeneric;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            // provide name for it
            auto allowFuncGenContext = GenContext(genContext);
            allowFuncGenContext.clearScopeVars();
            // if we set it to value we will not capture 'this' references
            allowFuncGenContext.thisType = nullptr;
            auto [result, funcOpRet, funcNameRet, isGenericRet] =
                mlirGenFunctionLikeDeclaration(arrowFunctionAST, allowFuncGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            funcOp = funcOpRet;
            funcName = funcNameRet;
            isGeneric = isGenericRet;
        }

        // if funcOp is null, means lambda is generic
        if (!funcOp)
        {
            // return reference to generic method
            if (getGenericFunctionMap().count(funcName))
            {
                auto genericFunctionInfo = getGenericFunctionMap().lookup(funcName);

                auto funcType = genericFunctionInfo->funcType ? genericFunctionInfo->funcType : getFunctionType({}, {}, false);

                // info: it will not take any capture now
                return resolveFunctionWithCapture(location, genericFunctionInfo->name, funcType,
                                                  mlir::Value(), true, genContext);
            }
            else
            {
                emitError(location) << "can't find generic function: " << funcName;
                return mlir::failure();
            }
        }

        assert(funcOp);

        return resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getFunctionType(), mlir::Value(), isGeneric, genContext);
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> mlirGenFunctionGenerator(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        auto fixThisReference = functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration;
        if (functionLikeDeclarationBaseAST->parameters.size() > 0)
        {
            auto nameNode = functionLikeDeclarationBaseAST->parameters.front()->name;
            if (nameNode == SyntaxKind::Identifier)
            {
                auto ident = nameNode.as<Identifier>();
                if (ident->escapedText == S(THIS_NAME))
                {
                    fixThisReference = true;
                }
            }
        }
        
        NodeFactory nf(NodeFactoryFlags::None);

        auto stepIdent = nf.createIdentifier(S(GENERATOR_STEP));

        // create return object
        NodeArray<ObjectLiteralElementLike> generatorObjectProperties;

        // add step field
        auto stepProp = nf.createPropertyAssignment(stepIdent, nf.createNumericLiteral(S("0"), TokenFlags::None));
        generatorObjectProperties.push_back(stepProp);

        // create body of next method
        NodeArray<Statement> nextStatements;

        // add main switcher
        auto stepAccess = nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), stepIdent);

        // call stateswitch
        auto callStat = nf.createExpressionStatement(
            nf.createCallExpression(nf.createIdentifier(S(GENERATOR_SWITCHSTATE)), undefined, {stepAccess}));

        nextStatements.push_back(callStat);

        // add function body to statements to first step
        if (functionLikeDeclarationBaseAST->body == SyntaxKind::Block)
        {
            // process every statement
            auto block = functionLikeDeclarationBaseAST->body.as<ts::Block>();
            for (auto statement : block->statements)
            {
                nextStatements.push_back(statement);
            }
        }
        else if (functionLikeDeclarationBaseAST->body)
        {
            nextStatements.push_back(functionLikeDeclarationBaseAST->body);
        }

        // add next statements
        // add default return with empty
        nextStatements.push_back(
            nf.createReturnStatement(getYieldReturnObject(nf, location, nf.createIdentifier(S(UNDEFINED_NAME)), true)));

        // create next body
        auto nextBody = nf.createBlock(nextStatements, /*multiLine*/ false);

        // create method next in object
        auto nextMethodDecl =
            nf.createMethodDeclaration(undefined, undefined, nf.createIdentifier(S(ITERATOR_NEXT)), undefined,
                                       undefined, undefined, undefined, nextBody);
        nextMethodDecl->internalFlags |= InternalFlags::VarsInObjectContext;

        // copy location info, to fix issue with names of anonymous functions
        nextMethodDecl->pos = functionLikeDeclarationBaseAST->pos;
        nextMethodDecl->_end = functionLikeDeclarationBaseAST->_end;

        if (fixThisReference)
        {
            FilterVisitorSkipFuncsAST<Node> visitor(SyntaxKind::ThisKeyword, [&](auto thisNode) {
                thisNode->internalFlags |= InternalFlags::ThisArgAlias;
            });

            for (auto it = begin(nextStatements) + 1; it != end(nextStatements); ++it)
            {
                visitor.visit(*it);
            }
        }

        generatorObjectProperties.push_back(nextMethodDecl);

        auto generatorObject = nf.createObjectLiteralExpression(generatorObjectProperties, false);

        // copy location info, to fix issue with names of anonymous functions
        generatorObject->pos = functionLikeDeclarationBaseAST->pos;
        generatorObject->_end = functionLikeDeclarationBaseAST->_end;

        // generator body
        NodeArray<Statement> generatorStatements;

        // TODO: this is hack, adding this as thisArg alias
        if (fixThisReference)
        {
            // TODO: this is temp hack, add this alias as thisArg, 
            NodeArray<VariableDeclaration> _thisArgDeclarations;
            auto _thisArg = nf.createIdentifier(S(THIS_ALIAS));
            _thisArgDeclarations.push_back(nf.createVariableDeclaration(_thisArg, undefined, undefined, nf.createToken(SyntaxKind::ThisKeyword)));
            auto _thisArgList = nf.createVariableDeclarationList(_thisArgDeclarations, NodeFlags::Const);

            generatorStatements.push_back(nf.createVariableStatement(undefined, _thisArgList));
        }

        // step 1, add return object
        auto retStat = nf.createReturnStatement(generatorObject);
        generatorStatements.push_back(retStat);

        auto body = nf.createBlock(generatorStatements, /*multiLine*/ false);

        if (functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            auto methodOp = nf.createMethodDeclaration(
                functionLikeDeclarationBaseAST->modifiers, undefined,
                functionLikeDeclarationBaseAST->name, undefined, functionLikeDeclarationBaseAST->typeParameters,
                functionLikeDeclarationBaseAST->parameters, functionLikeDeclarationBaseAST->type, body);

            // copy location info, to fix issue with names of anonymous functions
            methodOp->pos = functionLikeDeclarationBaseAST->pos;
            methodOp->_end = functionLikeDeclarationBaseAST->_end;        

            // to ensure correct full name
            methodOp->parent = functionLikeDeclarationBaseAST->parent;

            LLVM_DEBUG(printDebug(methodOp););

            auto genMethodOp = mlirGenFunctionLikeDeclaration(methodOp, genContext);
            return genMethodOp;            
        }
        else
        {
            auto funcOp = nf.createFunctionDeclaration(
                functionLikeDeclarationBaseAST->modifiers, undefined,
                functionLikeDeclarationBaseAST->name, functionLikeDeclarationBaseAST->typeParameters,
                functionLikeDeclarationBaseAST->parameters, functionLikeDeclarationBaseAST->type, body);

            // copy location info, to fix issue with names of anonymous functions
            funcOp->pos = functionLikeDeclarationBaseAST->pos;
            funcOp->_end = functionLikeDeclarationBaseAST->_end;        

            //LLVM_DEBUG(printDebug(funcOp););

            auto genFuncOp = mlirGenFunctionLikeDeclaration(funcOp, genContext);
            return genFuncOp;
        }
    }

    std::pair<mlir::LogicalResult, std::string> registerGenericFunctionLike(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, bool ignoreFunctionArgsDetection,
        const GenContext &genContext)
    {
        auto [fullName, name] = getNameOfFunction(functionLikeDeclarationBaseAST, genContext);
        if (name.empty())
        {
            return {mlir::failure(), name};
        }

        if (existGenericFunctionMap(name))
        {
            return {mlir::success(), name};
        }

        llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
        if (mlir::failed(
                processTypeParameters(functionLikeDeclarationBaseAST->typeParameters, typeParameters, genContext)))
        {
            return {mlir::failure(), name};
        }

        if (functionLikeDeclarationBaseAST->typeParameters.size() == 0) {
            processTypeParametersFromFunctionParameters(functionLikeDeclarationBaseAST, typeParameters, genContext);
        }

        // register class
        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = StringRef(fullName).copy(stringAllocator);
        GenericFunctionInfo::TypePtr newGenericFunctionPtr = std::make_shared<GenericFunctionInfo>();
        newGenericFunctionPtr->name = fullNamePtr;
        newGenericFunctionPtr->typeParams = typeParameters;
        newGenericFunctionPtr->functionDeclaration = functionLikeDeclarationBaseAST;
        newGenericFunctionPtr->elementNamespace = currentNamespace;
        newGenericFunctionPtr->typeParamsWithArgs = genContext.typeParamsWithArgs;

        // TODO: review it, ignore in case of ArrowFunction,
        if (!ignoreFunctionArgsDetection)
        {
            auto [result, funcOp] =
                getFuncArgTypesOfGenericMethod(functionLikeDeclarationBaseAST, typeParameters, false, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), name};
            }

            newGenericFunctionPtr->funcOp = funcOp;
            newGenericFunctionPtr->funcType = funcOp->getFuncType();

            LLVM_DEBUG(llvm::dbgs() << "\n!! registered generic function: " << name
                                    << ", type: " << funcOp->getFuncType() << "\n";);
        }

        getGenericFunctionMap().insert({namePtr, newGenericFunctionPtr});
        fullNameGenericFunctionsMap.insert(fullNamePtr, newGenericFunctionPtr);

        return {mlir::success(), name};
    }

    bool registerFunctionOp(FunctionPrototypeDOM::TypePtr funcProto, mlir_ts::FuncOp funcOp)
    {
        auto name = funcProto->getNameWithoutNamespace();
        if (!getFunctionMap().count(name))
        {
            getFunctionMap().insert({name, funcOp});

            LLVM_DEBUG(llvm::dbgs() << "\n!! reg. func: " << name << " type:" << funcOp.getFunctionType() << " function name: " << funcProto->getName()
                                    << " num inputs:" << mlir::cast<mlir_ts::FunctionType>(funcOp.getFunctionType()).getNumInputs()
                                    << "\n";);

            return true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! re-reg. func: " << name << " type:" << funcOp.getFunctionType() << " function name: " << funcProto->getName()
                                << " num inputs:" << mlir::cast<mlir_ts::FunctionType>(funcOp.getFunctionType()).getNumInputs()
                                << "\n";);

        return false;
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> mlirGenFunctionLikeDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto funcDeclGenContext = GenContext(genContext);
                
        auto isGenericFunction = 
            functionLikeDeclarationBaseAST->typeParameters.size() > 0 
            || !genContext.isGlobalVarReceiver && isGenericParameters(functionLikeDeclarationBaseAST, genContext);
        if (isGenericFunction && !funcDeclGenContext.instantiateSpecializedFunction)
        {
            auto [result, name] = registerGenericFunctionLike(functionLikeDeclarationBaseAST, false, funcDeclGenContext);
            return {result, mlir_ts::FuncOp(), name, false};
        }

        // check if it is generator
        if (functionLikeDeclarationBaseAST->asteriskToken)
        {
            // this is generator, let's generate other function out of it
            return mlirGenFunctionGenerator(functionLikeDeclarationBaseAST, funcDeclGenContext);
        }

        // do not process generic functions more then 1 time
        auto checkIfCreated = isGenericFunction && funcDeclGenContext.instantiateSpecializedFunction;
        if (checkIfCreated)
        {
            auto [fullFunctionName, functionName] = getNameOfFunction(functionLikeDeclarationBaseAST, funcDeclGenContext);

            auto funcOp = lookupFunctionMap(functionName);
            if (funcOp && theModule.lookupSymbol(functionName) 
                || theModule.lookupSymbol(fullFunctionName))
            {
                return {mlir::success(), funcOp, functionName, false};
            }
        }

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        if (isGenericFunction)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(theModule.getBody());
        }

        auto location = loc(functionLikeDeclarationBaseAST);

        auto [funcOp, funcProto, result, isGeneric] =
            mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, funcDeclGenContext);
        if (mlir::failed(result))
        {
            // in case of ArrowFunction without params and receiver is generic function as well
            return {result, funcOp, "", false};
        }

        if (mlir::succeeded(result) && isGeneric)
        {
            auto [result, name] = registerGenericFunctionLike(functionLikeDeclarationBaseAST, true, funcDeclGenContext);
            return {result, funcOp, name, isGeneric};
        }

        // check decorator for class
        auto dynamicImport = false;
        MLIRHelper::iterateDecorators(functionLikeDeclarationBaseAST, [&](std::string name, SmallVector<std::string> args) {
            if (name == DLL_IMPORT && args.size() > 0)
            {
                dynamicImport = true;
            }
        });

        if (dynamicImport)
        {
            // TODO: we do not need to register funcOp as we need to reference global variables
            auto result = mlirGenFunctionLikeDeclarationDynamicImport(
                location, funcProto->getNameWithoutNamespace(), funcOp.getFunctionType(), 
                funcProto->getName(), funcDeclGenContext, false);
            return {result, funcOp, funcProto->getName().str(), false};
        }

        auto funcGenContext = GenContext(funcDeclGenContext);
        funcGenContext.clearScopeVars();
        funcGenContext.funcOp = funcOp;
        funcGenContext.state = new int(1);
        // if funcGenContext.passResult is null and allocateVarsInContextThis is true, this type should contain fully
        // defined object with local variables as fields
        funcGenContext.allocateVarsInContextThis =
            (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::VarsInObjectContext) ==
            InternalFlags::VarsInObjectContext;

        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it != getCaptureVarsMap().end())
        {
            funcGenContext.capturedVars = &it->getValue();

            LLVM_DEBUG(llvm::dbgs() << "\n!! func has captured vars: " << funcProto->getName() << "\n";);
        }
        else
        {
            assert(funcGenContext.capturedVars == nullptr);
        }

        // register function to be able to call it if used in recursive call
        registerFunctionOp(funcProto, funcOp);

        // generate body
        auto resultFromBody = mlir::failure();
        {
            MLIRNamespaceGuard nsGuard(currentNamespace);
            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            SymbolTableScopeT varScope(symbolTable);
            resultFromBody = mlirGenFunctionBody(
                functionLikeDeclarationBaseAST, funcProto->getNameWithoutNamespace(), funcOp, funcProto, funcGenContext);
        }

        funcGenContext.cleanState();

        if (mlir::failed(resultFromBody))
        {
            return {mlir::failure(), funcOp, "", false};
        }

        // set visibility index
        auto isPublic = getExportModifier(functionLikeDeclarationBaseAST)
            || ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllExport) == InternalFlags::DllExport)
            /* we need to forcebly set to Public to prevent SymbolDCEPass to remove unsed name */
            || hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ExportKeyword)
            || ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::IsPublic) == InternalFlags::IsPublic)
            || funcProto->getName() == MAIN_ENTRY_NAME;
        if (isPublic)
        {
            funcOp.setPublic();
        }
        else
        {
            funcOp.setPrivate();
        }

        if (declarationMode && !funcDeclGenContext.dummyRun && funcProto->getNoBody())
        {
            funcOp.setPrivate();
        }

        if (!funcDeclGenContext.dummyRun)
        {
            theModule.push_back(funcOp);
        }

        if (isGenericFunction)
        {
            builder.restoreInsertionPoint(savePoint);
        }
        else
        {
            builder.setInsertionPointAfter(funcOp);
        }

        return {mlir::success(), funcOp, funcProto->getName().str(), false};
    }

    mlir::LogicalResult mlirGenStaticFieldDeclarationDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, StringRef name, mlir::Type type, const GenContext &genContext)
    {
        auto &staticFieldInfos = newClassPtr->staticFields;

        auto fieldId = MLIRHelper::TupleFieldName(name, builder.getContext());

        // register global
        auto fullClassStaticFieldName = concat(newClassPtr->fullName, name);

        auto staticFieldType =  mlir_ts::RefType::get(type);

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            // prevent double generating
            registerVariable(
                location, fullClassStaticFieldName, true, VariableType::Var,
                [&](mlir::Location location, const GenContext &genContext)  -> TypeValueInitType {
                    auto fullName = V(mlirGenStringValue(location, fullClassStaticFieldName.str(), true));
                    auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, getOpaqueType(), fullName);
                    auto result = cast(location, staticFieldType, referenceToStaticFieldOpaque, genContext);
                    auto referenceToStaticField = V(result);
                    return {referenceToStaticField.getType(), referenceToStaticField, TypeProvided::Yes};
                },
                genContext);
        }

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1);

        return mlir::success();
    }    

    mlir::LogicalResult mlirGenFunctionLikeDeclarationDynamicImport(mlir::Location location, StringRef funcName, mlir_ts::FunctionType functionType, StringRef dllFuncName, const GenContext &genContext, bool isFullNamespaceName = true)
    {
        registerVariable(location, funcName, isFullNamespaceName, VariableType::Var,
            [&](mlir::Location location, const GenContext &context) -> TypeValueInitType {
                // add command to load reference fron DLL
                auto fullName = V(mlirGenStringValue(location, dllFuncName.str(), true));
                auto referenceToFuncOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, getOpaqueType(), fullName);
                auto result = cast(location, functionType, referenceToFuncOpaque, genContext);
                auto referenceToFunc = V(result);
                return {referenceToFunc.getType(), referenceToFunc, TypeProvided::No};
            },
            genContext);

        return mlir::success();
    }    

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto,
                                             const GenContext &genContext)
    {
        return mlirGenFunctionEntry(location, funcProto->getReturnType(), genContext);
    }

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, mlir::Type retType, const GenContext &genContext)
    {
        auto hasReturn = retType && !isa<mlir_ts::VoidType>(retType);
        if (hasReturn)
        {
            auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
            auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
            varDecl->setReadWriteAccess();
            DECLARE(varDecl, entryOp.getReference());
        }
        else
        {
            builder.create<mlir_ts::EntryOp>(location, mlir::Type());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionExit(mlir::Location location, const GenContext &genContext)
    {
        auto callableResult = const_cast<GenContext &>(genContext).funcOp.getCallableResults();
        auto retType = callableResult.size() > 0 ? callableResult.front() : mlir::Type();
        auto hasReturn = retType && !isa<mlir_ts::VoidType>(retType);
        if (hasReturn)
        {
            auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
            if (!retVarInfo.second)
            {
                if (genContext.allowPartialResolve)
                {
                    return mlir::success();
                }

                emitError(location) << "can't find return variable";
                return mlir::failure();
            }

            builder.create<mlir_ts::ExitOp>(location, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ExitOp>(location, mlir::Value());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCapturedParam(mlir::Location location, int &firstIndex,
                                                     FunctionPrototypeDOM::TypePtr funcProto,
                                                     mlir::Block::BlockArgListType arguments,
                                                     const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto isObjectType = genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
        if (isObjectType)
        {
            return mlir::success();
        }

        auto capturedParam = arguments[firstIndex++];
        auto capturedRefType = capturedParam.getType();

        auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, capturedRefType, location);

        DECLARE(capturedParamVar, capturedParam);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCapturedParamIfObject(mlir::Location location, int &firstIndex,
                                                             FunctionPrototypeDOM::TypePtr funcProto,
                                                             mlir::Block::BlockArgListType arguments,
                                                             const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto isObjectType = genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
        if (isObjectType)
        {

            auto thisVal = resolveIdentifier(location, THIS_NAME, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this value: " << thisVal << "\n";);

            auto capturedNameResult =
                mlirGenPropertyAccessExpression(location, thisVal, MLIRHelper::TupleFieldName(CAPTURED_NAME, builder.getContext()), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(capturedNameResult)

            mlir::Value propValue = V(capturedNameResult);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this->.captured value: " << propValue << "\n";);

            assert(propValue);

            // captured is in this->".captured"
            auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, propValue.getType(), location);
            DECLARE(capturedParamVar, propValue);
        }

        return mlir::success();
    }

    // TODO: put into MLIRCodeLogicHelper
    ValueOrLogicalResult optionalValueOrUndefinedExpression(mlir::Location location, mlir::Value condValue, Expression expression, const GenContext &genContext)
    {
        return optionalValueOrUndefined(location, condValue, [&](auto genContext) { return mlirGen(expression, genContext); }, genContext);
    }

    // TODO: put into MLIRCodeLogicHelper
    ValueOrLogicalResult optionalValueOrUndefined(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult(const GenContext &)> exprFunc, const GenContext &genContext)
    {
        return conditionalValue(location, condValue, 
            [&](auto genContext) { 
                auto result = exprFunc(genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto value = V(result);
                auto optValue = 
                    isa<mlir_ts::OptionalType>(value.getType())
                        ? value
                        : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                return ValueOrLogicalResult(optValue); 
            }, 
            [&](mlir::Type trueValueType, auto genContext) { 
                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, trueValueType);
                return ValueOrLogicalResult(optUndefValue); 
            }, 
            genContext);
    }

    // TODO: put into MLIRCodeLogicHelper
    ValueOrLogicalResult anyOrUndefined(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult(const GenContext &)> exprFunc, const GenContext &genContext)
    {
        return conditionalValue(location, condValue, 
            [&](auto genContext) { 
                auto result = exprFunc(genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto value = V(result);
                auto anyValue = V(builder.create<mlir_ts::CastOp>(location, getAnyType(), value));
                return ValueOrLogicalResult(anyValue); 
            }, 
            [&](mlir::Type trueValueType, auto genContext) {
                auto undefValue = builder.create<mlir_ts::UndefOp>(location, getUndefinedType());
                auto anyUndefValue = V(builder.create<mlir_ts::CastOp>(location, trueValueType, undefValue));
                return ValueOrLogicalResult(anyUndefValue); 
            }, 
            genContext);
    }

    // TODO: put into MLIRCodeLogicHelper
    // TODO: we have a lot of IfOp - create 1 logic for conditional values
    ValueOrLogicalResult conditionalValue(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult(const GenContext &)> trueValue, 
        std::function<ValueOrLogicalResult(mlir::Type trueValueType, const GenContext &)> falseValue, 
        const GenContext &genContext)
    {
        // type will be set later
        auto ifOp = builder.create<mlir_ts::IfOp>(location, builder.getNoneType(), condValue, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

        // value if true
        auto trueResult = trueValue(genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(trueResult)
        ifOp.getResults().front().setType(trueResult.value.getType());
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{trueResult});

        // else
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

        // value if false
        auto falseResult = falseValue(trueResult.value.getType(), genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(falseResult)
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{falseResult});

        builder.setInsertionPointAfter(ifOp);

        return ValueOrLogicalResult(ifOp.getResults().front());        
    }    

    // TODO: put into MLIRCodeLogicHelper
    ValueOrLogicalResult optionalValueOrDefault(mlir::Location location, mlir::Type dataType, mlir::Value value, Expression defaultExpr, const GenContext &genContext)
    {
        auto optionalValueOrDefaultOp = builder.create<mlir_ts::OptionalValueOrDefaultOp>(
            location, dataType, value);

        /*auto *defValueBlock =*/builder.createBlock(&optionalValueOrDefaultOp.getDefaultValueRegion());

        mlir::Value defaultValue;
        if (defaultExpr)
        {
            auto result = mlirGen(defaultExpr, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result);
            defaultValue = V(result);
        }
        else
        {
            llvm_unreachable("unknown statement");
        }

        if (defaultValue.getType() != dataType)
        {
            CAST(defaultValue, location, dataType, defaultValue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, defaultValue);

        builder.setInsertionPointAfter(optionalValueOrDefaultOp);

        return V(optionalValueOrDefaultOp);
    } 

    ValueOrLogicalResult processOptionalParam(mlir::Location location, int index, mlir::Type dataType, mlir::Value value, Expression defaultExpr, const GenContext &genContext)
    {
        auto paramOptionalOp = builder.create<mlir_ts::ParamOptionalOp>(
            location, mlir_ts::RefType::get(dataType), value, builder.getBoolAttr(false), builder.getIndexAttr(index + 1));

        /*auto *defValueBlock =*/builder.createBlock(&paramOptionalOp.getDefaultValueRegion());

        mlir::Value defaultValue;
        if (defaultExpr)
        {
            auto result = mlirGen(defaultExpr, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result);
            defaultValue = V(result);
        }
        else
        {
            llvm_unreachable("unknown statement");
        }

        if (defaultValue.getType() != dataType)
        {
            CAST(defaultValue, location, dataType, defaultValue, genContext);
        }

        builder.create<mlir_ts::ParamDefaultValueOp>(location, defaultValue);

        builder.setInsertionPointAfter(paramOptionalOp);

        return V(paramOptionalOp);
    }    

    mlir::LogicalResult mlirGenFunctionParams(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        for (auto [paramIndex, param] : enumerate(funcProto->getParams()))
        {
            auto index = firstIndex + (int)paramIndex;
            mlir::Value paramValue;

            // process init expression
            // we need reset scope for location as location of funcProto was created before correct scope
            auto location = locFuseWithScope(stripMetadata(param->getLoc()));

            LLVM_DEBUG(llvm::dbgs() << "Location for Param: " << location << "\n");

            // alloc all args
            // process optional parameters
            if (param->hasInitValue())
            {
                auto result = processOptionalParam(location, index, param->getType(), arguments[index], param->getInitValue(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                paramValue = V(result);
            }
            else if (param->getIsOptional() && !isa<mlir_ts::OptionalType>(param->getType()))
            {
                auto optType = getOptionalType(param->getType());
                param->setType(optType);
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(optType),
                        arguments[index], builder.getBoolAttr(false), builder.getIndexAttr(index + 1));
            }
            else
            {
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(param->getType()),
                        arguments[index], builder.getBoolAttr(false), builder.getIndexAttr(index + 1));
            }

            if (paramValue)
            {
                // redefine variable
                param->setReadWriteAccess();
                DECLARE(param, paramValue);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionParams(mlir::Location location, int firstIndex, mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        for (auto index = firstIndex; index < arguments.size(); index++)
        {
            std::string paramName("p");
            paramName += std::to_string(index - firstIndex);
            
            auto paramDecl = std::make_shared<VariableDeclarationDOM>(paramName, arguments[index].getType(), location);        
            DECLARE(paramDecl, arguments[index]);
        }

        return mlir::success();
    }    

    mlir::LogicalResult mlirGenFunctionParamsBindings(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                      mlir::Block::BlockArgListType arguments,
                                                      const GenContext &genContext)
    {
        for (const auto &param : funcProto->getParams())
        {
            if (auto bindingPattern = param->getBindingPattern())
            {
                auto location = loc(bindingPattern);
                auto val = resolveIdentifier(location, param->getName(), genContext);
                assert(val);
                auto initFunc = [&](mlir::Location, const GenContext &) { return std::make_tuple(val.getType(), val, TypeProvided::No); };

                if (bindingPattern == SyntaxKind::ArrayBindingPattern)
                {
                    auto arrayBindingPattern = bindingPattern.as<ArrayBindingPattern>();
                    if (mlir::failed(processDeclarationArrayBindingPattern(location, arrayBindingPattern, VariableType::Let,
                                                               initFunc, genContext)))
                    {
                        return mlir::failure();
                    }
                }
                else if (bindingPattern == SyntaxKind::ObjectBindingPattern)
                {
                    auto objectBindingPattern = bindingPattern.as<ObjectBindingPattern>();
                    if (mlir::failed(processDeclarationObjectBindingPattern(location, objectBindingPattern, VariableType::Let,
                                                                initFunc, genContext)))
                    {
                        return mlir::failure();
                    }
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCaptures(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto capturedVars = *genContext.capturedVars;

        NodeFactory nf(NodeFactoryFlags::None);

        // create variables
        for (auto &capturedVar : capturedVars)
        {
            auto varItem = capturedVar.getValue();
            auto variableInfo = varItem;
            auto name = variableInfo->getName();

            // load this.<var name>
            auto _captured = nf.createIdentifier(stows(CAPTURED_NAME));
            auto _name = nf.createIdentifier(stows(std::string(name)));
            auto _captured_name = nf.createPropertyAccessExpression(_captured, _name);
            auto result = mlirGen(_captured_name, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto capturedVarValue = V(result);
            auto variableRefType = mlir_ts::RefType::get(variableInfo->getType());

            auto capturedParam =
                std::make_shared<VariableDeclarationDOM>(name, variableRefType, variableInfo->getLoc());
            assert(capturedVarValue);
            if (isa<mlir_ts::RefType>(capturedVarValue.getType()))
            {
                capturedParam->setReadWriteAccess();
            }

            LLVM_DEBUG(dbgs() << "\n!! captured '\".captured\"->" << name << "' [ " << capturedVarValue
                              << " ] ref val type: [ " << variableRefType << " ]");

            DECLARE(capturedParam, capturedVarValue);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                            StringRef name, mlir_ts::FuncOp funcOp, FunctionPrototypeDOM::TypePtr funcProto,
                                            const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> FUNCTION: '" << funcProto->getName() << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        if (!functionLikeDeclarationBaseAST->body || declarationMode && !genContext.dummyRun)
        {
            // it is just declaration
            funcProto->setNoBody(true);
            return mlir::success();
        }

        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(functionLikeDeclarationBaseAST);

        // Debug Info
        DITableScopeT debugFuncScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            auto locWithDI = 
                mdi.getSubprogram(
                    location, 
                    name,
                    funcOp.getName(), 
                    functionLikeDeclarationBaseAST->body 
                        ? loc(functionLikeDeclarationBaseAST->body) 
                        : location);

            LLVM_DEBUG(llvm::dbgs() << "Location of func: " << locWithDI << "\n");

            funcOp->setLoc(locWithDI);
        }

        // new location withing FunctionScope
        location = loc(functionLikeDeclarationBaseAST->body);

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();
        auto firstIndex = 0;

        // add exit code
        if (failed(mlirGenFunctionEntry(location, funcProto, genContext)))
        {
            return mlir::failure();
        }

        // register this if lambda function
        if (failed(mlirGenFunctionCapturedParam(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionParamsBindings(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCapturedParamIfObject(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCaptures(location, funcProto, genContext)))
        {
            return mlir::failure();
        }

        // if we need params only we do not need to process body
        auto discoverParamsOnly = genContext.allowPartialResolve && genContext.discoverParamsOnly;
        if (!discoverParamsOnly)
        {
            // we need it to skip lexical block
            functionLikeDeclarationBaseAST->body->parent = functionLikeDeclarationBaseAST->body;
            if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, genContext)))
            {
                return mlir::failure();
            }
        }

        // add exit code
        if (failed(mlirGenFunctionExit(location, genContext)))
        {
            return mlir::failure();
        }

        if (genContext.dummyRun && genContext.cleanUps)
        {
            genContext.cleanUps->push_back(blockPtr);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> FUNCTION (SUCCESS END): '" << funcProto->getName() << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionBody(mlir::Location location, StringRef funcName, StringRef fullFuncName,
                                            mlir_ts::FunctionType funcType, std::function<mlir::LogicalResult(mlir::Location, const GenContext &)> funcBody,                                            
                                            const GenContext &genContext,
                                            int firstParam = 0, bool isPublic = false)
    {
        if (theModule.lookupSymbol(fullFuncName))
        {
            return mlir::success();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> SYNTH. FUNCTION: '" << fullFuncName << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        SymbolTableScopeT varScope(symbolTable);

        auto funcOp = mlir_ts::FuncOp::create(location, fullFuncName, funcType);

        // Debug Info
        DITableScopeT debugFuncScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            auto locWithDI = 
                mdi.getSubprogram(
                    location, 
                    funcName,
                    fullFuncName, 
                    location);
            funcOp->setLoc(locWithDI);

            // new location withing FunctionScope
            location = locFuseWithScope(stripMetadata(location));
        }

        GenContext funcGenContext(genContext);
        funcGenContext.funcOp = funcOp;

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();

        // add exit code
        if (failed(mlirGenFunctionEntry(location, mth.getReturnTypeFromFuncRef(funcType), funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionParams(location, firstParam, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(funcBody(location, funcGenContext)))
        {
            return mlir::failure();
        }

        // add exit code
        auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
        if (retVarInfo.first)
        {
            builder.create<mlir_ts::ExitOp>(location, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ExitOp>(location, mlir::Value());
        }

        if (genContext.dummyRun)
        {
            if (genContext.cleanUps)
            {
                genContext.cleanUps->push_back(blockPtr);
            }
        }
        else
        {
            theModule.push_back(funcOp);
        }

        if (isPublic)
        {
            funcOp.setPublic();
        }
        else
        {
            funcOp.setPrivate();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> SYNTH. FUNCTION (SUCCESS END): '" << fullFuncName << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext)
    {
        auto location = loc(typeAssertionAST);

        auto typeInfo = getType(typeAssertionAST->type, genContext);
        if (!typeInfo)
        {
            return mlir::failure();
        }

        GenContext noReceiverGenContext(genContext);
        noReceiverGenContext.clearReceiverTypes();
        noReceiverGenContext.receiverType = typeInfo;

        auto result = mlirGen(typeAssertionAST->expression, noReceiverGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        CAST_A(castedValue, location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    ValueOrLogicalResult mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
    {
        auto location = loc(asExpressionAST);

        auto typeInfo = getType(asExpressionAST->type, genContext);
        if (!typeInfo)
        {
            return mlir::failure();
        }

        GenContext noReceiverGenContext(genContext);
        noReceiverGenContext.clearReceiverTypes();
        noReceiverGenContext.receiverType = typeInfo;

        auto result = mlirGen(asExpressionAST->expression, noReceiverGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        CAST_A(castedValue, location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    ValueOrLogicalResult mlirGen(ComputedPropertyName computedPropertyNameAST, const GenContext &genContext)
    {
        auto result = mlirGen(computedPropertyNameAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);
        return exprValue;
    }

    mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
    {
        auto location = loc(returnStatementAST);
        if (auto expression = returnStatementAST->expression)
        {
            GenContext receiverTypeGenContext(genContext);
            receiverTypeGenContext.clearReceiverTypes();
            auto exactReturnType = getExplicitReturnTypeOfCurrentFunction(genContext);
            if (exactReturnType)
            {
                receiverTypeGenContext.receiverType = exactReturnType;
            }

            auto result = mlirGen(expression, receiverTypeGenContext);
            EXIT_IF_FAILED(result)
            
            auto expressionValue = V(result);
            if (!expressionValue)
            {
                emitError(location, "No return value");
            }

            if (!genContext.allowPartialResolve)
            {
                VALIDATE(expressionValue, location)
            }

            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::FullStack, {}, &genContext));

            return mlirGenReturnValue(location, expressionValue, false, genContext);
        }

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::FullStack, {}, &genContext));

        builder.create<mlir_ts::ReturnOp>(location);
        return mlir::success();
    }

    ObjectLiteralExpression getYieldReturnObject(NodeFactory &nf, mlir::Location location, Expression expr, bool stop)
    {
        auto valueIdent = nf.createIdentifier(S("value"));
        auto doneIdent = nf.createIdentifier(S("done"));

        NodeArray<ObjectLiteralElementLike> retObjectProperties;
        auto valueProp = nf.createPropertyAssignment(valueIdent, expr);
        retObjectProperties.push_back(valueProp);

        auto doneProp = nf.createPropertyAssignment(
            doneIdent, nf.createToken(stop ? SyntaxKind::TrueKeyword : SyntaxKind::FalseKeyword));
        retObjectProperties.push_back(doneProp);

        auto retObject = nf.createObjectLiteralExpression(retObjectProperties, stop);
        
        // copy location info, to fix issue with names of anonymous functions
        LocationHelper lh(builder.getContext());
        auto [pos, _end] = lh.getSpan(location);

        assert(pos != _end && pos > 0);

        retObject->pos = pos;
        retObject->_end = _end;        

        return retObject;
    };

    ValueOrLogicalResult mlirGenYieldStar(YieldExpression yieldExpressionAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _v_ident = nf.createIdentifier(S(".v"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        auto _yield_expr = nf.createYieldExpression(undefined, _v_ident);
        // copy location info, to fix issue with names of anonymous functions
        _yield_expr->pos = yieldExpressionAST->pos;
        _yield_expr->_end = yieldExpressionAST->_end;

        auto forOfStat =
            nf.createForOfStatement(undefined, declList, yieldExpressionAST->expression,
                                    nf.createExpressionStatement(_yield_expr));

        return mlirGen(forOfStat, genContext);
    }

    ValueOrLogicalResult mlirGen(YieldExpression yieldExpressionAST, const GenContext &genContext)
    {
        if (yieldExpressionAST->asteriskToken)
        {
            return mlirGenYieldStar(yieldExpressionAST, genContext);
        }

        auto location = loc(yieldExpressionAST);

        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        // get state
        auto state = 0;
        if (genContext.state)
        {
            state = (*genContext.state)++;
        }
        else
        {
            assert(false);
        }

        // set restore point (return point)
        stringstream num;
        num << state;

        NodeFactory nf(NodeFactoryFlags::None);

        if (evaluateProperty(nf.createToken(SyntaxKind::ThisKeyword), GENERATOR_STEP, genContext))
        {
            // save return point - state -> this.step = xxx
            auto setStateExpr = nf.createBinaryExpression(
                nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(S(GENERATOR_STEP))),
                nf.createToken(SyntaxKind::EqualsToken), nf.createNumericLiteral(num.str(), TokenFlags::None));
            mlirGen(setStateExpr, genContext);
        }
        else
        {
            // save return point - state -> step = xxx
            auto setStateExpr = nf.createBinaryExpression(
                nf.createIdentifier(S(GENERATOR_STEP)),
                nf.createToken(SyntaxKind::EqualsToken), nf.createNumericLiteral(num.str(), TokenFlags::None));
            mlirGen(setStateExpr, genContext);
        }

        // return value
        auto yieldRetValue = getYieldReturnObject(nf, location, yieldExpressionAST->expression, false);
        auto result = mlirGen(yieldRetValue, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto yieldValue = V(result);

        mlirGenReturnValue(location, yieldValue, true, genContext);

        std::stringstream label;
        label << GENERATOR_STATELABELPREFIX << state;
        builder.create<mlir_ts::StateLabelOp>(location, label.str());

        // TODO: yield value to continue, should be loaded from "next(value)" parameter
        // return yieldValue;
        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(AwaitExpression awaitExpressionAST, const GenContext &genContext)
    {
#ifdef ENABLE_ASYNC
        // TODO: due to cloning code into next function, it is not fixing scope properly
        auto location = stripMetadata(loc(awaitExpressionAST));

        auto resultType = evaluate(awaitExpressionAST->expression, genContext);

        ValueOrLogicalResult result(mlir::failure());
        auto asyncExecOp = builder.create<mlir::async::ExecuteOp>(
            location, resultType ? mlir::TypeRange{resultType} : mlir::TypeRange(), mlir::ValueRange{},
            mlir::ValueRange{}, [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                DITableScopeT debugAsyncCodeScope(debugScope);
                MLIRDebugInfoHelper mdi(builder, debugScope);

                // TODO: temp hack to break wrong chain on scopes because 'await' create extra function wrap
                mdi.clearDebugScope();
                mdi.setLexicalBlock(location);

                result = mlirGen(awaitExpressionAST->expression, genContext);
                if (result)
                {
                    auto value = V(result);
                    if (value)
                    {
                        builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{value});
                    }
                    else
                    {
                        builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                    }
                }
            });
        EXIT_IF_FAILED_OR_NO_VALUE(result)

        if (resultType)
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.getResults().back());
            return asyncAwaitOp.getResult();
        }
        else
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.getToken());
        }

        return mlir::success();
#else
        return mlirGen(awaitExpressionAST->expression, genContext);
#endif
    }

    mlir::LogicalResult processReturnType(mlir::Location location, mlir::Value expressionValue, const GenContext &genContext)
    {
        // TODO: rewrite it using UnionType

        // record return type if not provided
        if (genContext.passResult)
        {
            if (!expressionValue)
            {
                return mlir::failure();
            }

            auto type = expressionValue.getType();
            LLVM_DEBUG(dbgs() << "\n!! processing return type: " << type << "");

            if (mth.isNoneType(type))
            {
                return mlir::success();
            }

            type = mth.wideStorageType(type);

            // if return type is not detected, take first and exit
            if (!genContext.passResult->functionReturnType)
            {
                genContext.passResult->functionReturnType = type;
                return mlir::success();
            }

            // TODO: undefined & null should be processed as union type
            auto undefType = getUndefinedType();
            auto nullType = getNullType();

            // filter out types, such as: undefined, objects with undefined values etc
            if (type == undefType || type == nullType)
            {
                return mlir::failure();
            }

            // if (mth.hasUndefines(type))
            // {
            //     return mlir::failure();
            // }

            auto merged = false;
            auto resultReturnType = mth.mergeType(location, genContext.passResult->functionReturnType, type, merged);            

            LLVM_DEBUG(dbgs() << "\n!! return type: " << resultReturnType << "");

            genContext.passResult->functionReturnType = resultReturnType;
        }

        return mlir::success();
    }

    mlir::Type getExplicitReturnTypeOfCurrentFunction(const GenContext &genContext)
    {
        auto funcOp = const_cast<GenContext &>(genContext).funcOp;
        if (funcOp)
        {
            auto countResults = funcOp.getCallableResults().size();
            if (countResults > 0)
            {
                auto returnType = funcOp.getCallableResults().front();
                return returnType;
            }
        }

        return mlir::Type();
    }

    mlir::LogicalResult mlirGenReturnValue(mlir::Location location, mlir::Value expressionValue, bool yieldReturn,
                                           const GenContext &genContext)
    {
        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        if (auto returnType = getExplicitReturnTypeOfCurrentFunction(genContext))
        {
            if (!expressionValue)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "'return' must have value";
                    return mlir::failure();
                }
            }
            else if (returnType != expressionValue.getType())
            {
                CAST_A(castValue, location, returnType, expressionValue, genContext);
                expressionValue = castValue;
            }
        }

        // record return type if not provided
        processReturnType(location, expressionValue, genContext);

        if (!expressionValue)
        {
            emitError(location) << "'return' must have value";
            builder.create<mlir_ts::ReturnOp>(location);
            return genContext.passResult ? mlir::success() : mlir::failure();
        }

        auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
        if (!retVarInfo.second)
        {
            if (genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            emitError(location) << "can't find return variable";
            return mlir::failure();
        }

        if (yieldReturn)
        {
            builder.create<mlir_ts::YieldReturnValOp>(location, expressionValue, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);
        }

        return mlir::success();
    }

    struct ElseSafeCase
    {
        Expression expr;
        mlir::Type safeType;
    };

    mlir::LogicalResult addSafeCastStatement(Expression expr, Node typeToken, bool inverse, ElseSafeCase* elseSafeCase, const GenContext &genContext)
    {
        auto safeType = getType(typeToken, genContext);
        return addSafeCastStatement(expr, safeType, inverse, elseSafeCase, genContext);
    }

    bool isSafeTypeTheSameAndNoNeedToCast(mlir::Type type, mlir::Type safeType)
    {
        if (type == safeType)
        {
            // nothing todo here
            return true;
        }
        else if (mlir::isa<mlir_ts::ArrayType>(safeType) && mlir::isa<mlir_ts::ArrayType>(type))
        {
            // nothing todo here
            return true;
        }

        return false;
    }

    mlir::LogicalResult addSafeCastStatement(Expression expr, mlir::Type safeType, bool inverse, ElseSafeCase* elseSafeCase, const GenContext &genContext)
    {
        auto location = loc(expr);
        auto nameStr = MLIRHelper::getName(expr.as<DeclarationName>());
        auto result = mlirGen(expr, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result);
        auto exprValue = V(result);

        if (isSafeTypeTheSameAndNoNeedToCast(exprValue.getType(), safeType))
        {
            return mlir::success();
        }

        if (elseSafeCase)
        {
            elseSafeCase->expr = expr;
        }

        return addSafeCastStatement(location, nameStr, exprValue, safeType, inverse, elseSafeCase, genContext);
    }    

    mlir::LogicalResult addSafeCastStatement(mlir::Location location, std::string parameterName, mlir::Value exprValue, mlir::Type safeType, bool inverse, ElseSafeCase* elseSafeCase, const GenContext &genContext)
    {
        mlir::Value castedValue;
        if (isa<mlir_ts::AnyType>(exprValue.getType()))
        {
            if (inverse) return mlir::failure();
            castedValue = builder.create<mlir_ts::UnboxOp>(location, safeType, exprValue);
        }
        else if (isa<mlir_ts::OptionalType>(exprValue.getType()) 
                 && mlir::cast<mlir_ts::OptionalType>(exprValue.getType()).getElementType() == safeType)
        {
            if (inverse) 
            {
                if (elseSafeCase)
                {
                    // it will be process in "else" clause
                    elseSafeCase->safeType = safeType;
                }

                return mlir::failure();
            }
            else
            {
                castedValue = builder.create<mlir_ts::ValueOp>(location, safeType, exprValue);
            }
        }
        else if (auto unionType = dyn_cast<mlir_ts::UnionType>(exprValue.getType()))
        {
            // prepare else case first
            if (elseSafeCase)
            {
                // add else case
                auto types = unionType.getTypes();
                SmallVector<mlir::Type> newTypes;
                for (auto& subUnionType : types)
                {
                    if (inverse && subUnionType != safeType) continue;
                    if (!inverse && subUnionType == safeType) continue;
                    newTypes.push_back(subUnionType);
                }

                elseSafeCase->safeType = getUnionType(newTypes);
            }

            if (!inverse)
            {
                if (isa<mlir_ts::UnionType>(safeType))
                {
                    // no need to cast union type <type1 | type2 | type3 > to <type 1 | type 3> as it will be
                    // the same LLVMType structure
                    //return mlir::failure();
                    // in case of union we just want to have the same structured type but with less types in union
                    castedValue = builder.create<mlir_ts::CastOp>(location, safeType, exprValue);
                }
                else
                {
                    castedValue = builder.create<mlir_ts::GetValueFromUnionOp>(location, safeType, exprValue);
                }
            }
            else
            {
                auto types = unionType.getTypes();
                SmallVector<mlir::Type> newTypes;
                for (auto& subUnionType : types)
                {
                    if (subUnionType == safeType) continue;
                    newTypes.push_back(subUnionType);
                }

                auto newSafeType = getUnionType(newTypes);

                return addSafeCastStatement(location, parameterName, exprValue, newSafeType, false, nullptr, genContext);
            }
        }
        else
        {
            if (inverse) return mlir::failure();
            CAST_A(result, location, safeType, exprValue, genContext);
            castedValue = V(result);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! Safe Type: [" << parameterName << "] is [" << safeType << "]\n");

        return 
            !!registerVariable(
                location, parameterName, false, VariableType::Const,
                [&](mlir::Location, const GenContext &) -> TypeValueInitType
                {
                    return {safeType, castedValue, TypeProvided::Yes};
                },
                genContext) ? mlir::success() : mlir::failure();        
    }    

    mlir::LogicalResult checkSafeCastTypeOf(Expression typeOfVal, Expression constVal, bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        if (typeOfVal == SyntaxKind::TypeOfExpression)
        {
            auto typeOfOp = typeOfVal.as<TypeOfExpression>();
            // strip parenthesizes
            auto expr = stripParentheses(typeOfOp->expression);
            if (expr != SyntaxKind::Identifier)
            {
                return mlir::failure();
            }

            if (auto stringLiteral = constVal.as<ts::StringLiteral>())
            {
                // create 'expression' = <string>'expression;
                NodeFactory nf(NodeFactoryFlags::None);

                auto text = stringLiteral->text;
                Node typeToken;
                if (text == S("boolean"))
                {
                    typeToken = nf.createToken(SyntaxKind::BooleanKeyword);
                }
                else if (text == S("number"))
                {
                    typeToken = nf.createToken(SyntaxKind::NumberKeyword);
                }
                else if (text == S("string"))
                {
                    typeToken = nf.createToken(SyntaxKind::StringKeyword);
                }
                else if (text == S("bigint"))
                {
                    typeToken = nf.createToken(SyntaxKind::BigIntKeyword);
                }
                else if (text == S("function") || text == S("class") || text == S("interface") || text == S("object"))
                {
                    typeToken = nf.createTypeReferenceNode(nf.createIdentifier(S("Opaque")));
                }
                else if (text == S("array")) 
                {
                    auto typeTokenElement = nf.createTypeReferenceNode(nf.createIdentifier(S("Opaque")));
                    typeToken = nf.createArrayTypeNode(typeTokenElement);
                }
                else if (isEmbededTypeWithBuiltins(wstos(text)))
                {
                    typeToken = nf.createTypeReferenceNode(nf.createIdentifier(text));
                }

                if (typeToken)
                {
                    return addSafeCastStatement(expr, typeToken, inverse, elseSafeCase, genContext);
                }

                return mlir::success();
            }
        }

        return mlir::failure();
    }

    mlir::LogicalResult checkSafeCastUndefined(Expression optVal, Expression undefVal, bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        auto expr = stripParentheses(undefVal);
        if (expr == SyntaxKind::Identifier)
        {
            auto identifier = expr.as<ts::Identifier>();
            if (identifier->escapedText == S(UNDEFINED_NAME))
            {
                auto optEval = evaluate(optVal, genContext);
                if (auto optType = dyn_cast_or_null<mlir_ts::OptionalType>(optEval))
                {
                    return addSafeCastStatement(optVal, optType.getElementType(), inverse, elseSafeCase, genContext);
                }
            }
        }

        return mlir::failure();
    }    

    mlir::LogicalResult checkSafeCastBoolean(Expression exprVal, bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        auto exprEval = evaluate(exprVal, genContext);
        if (auto optType = dyn_cast_or_null<mlir_ts::OptionalType>(exprEval))
        {
            return addSafeCastStatement(exprVal, optType.getElementType(), inverse, elseSafeCase, genContext);
        }

        return mlir::failure();
    }    

    Expression stripParentheses(Expression exprVal)
    {
        auto expr = exprVal;
        while (expr == SyntaxKind::ParenthesizedExpression)
        {
            expr = expr.as<ParenthesizedExpression>()->expression;
        }

        return expr;
    }

    mlir::LogicalResult checkSafeCastPropertyAccessLogic(TextRange textRange, Expression objAccessExpression,
                                                         mlir::Type typeOfObject, Node name, mlir::Value constVal,
                                                         bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(typeOfObject))
        {
            auto isConst = false;
            mlir::Attribute value;
            isConst = isConstValue(constVal);
            if (isConst)
            {
                auto constantOp = constVal.getDefiningOp<mlir_ts::ConstantOp>();
                assert(constantOp);
                auto valueAttr = constantOp.getValueAttr();

                MLIRCodeLogic mcl(builder);
                auto fieldNameAttr = TupleFieldName(name, genContext);

                for (auto unionSubType : unionType.getTypes())
                {
                    if (auto tupleType = dyn_cast<mlir_ts::TupleType>(unionSubType))
                    {
                        auto fieldIndex = tupleType.getIndex(fieldNameAttr);
                        auto fieldType = tupleType.getType(fieldIndex);
                        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(fieldType))
                        {
                            if (literalType.getValue() == valueAttr)
                            {
                                // enable safe cast found
                                auto typeAliasNameUTF8 = MLIRHelper::getAnonymousName(loc_check(textRange), "ta_", getNamespaceName());
                                auto typeAliasName = convertUTF8toWide(typeAliasNameUTF8);
                                const_cast<GenContext &>(genContext)
                                    .typeAliasMap.insert({typeAliasNameUTF8, tupleType});

                                NodeFactory nf(NodeFactoryFlags::None);
                                auto typeRef = nf.createTypeReferenceNode(nf.createIdentifier(typeAliasName));
                                return addSafeCastStatement(objAccessExpression, typeRef, inverse, elseSafeCase, genContext);
                            }
                        }
                    }

                    if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(unionSubType))
                    {
                        if (auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue()))
                        {
                            auto fieldInfo = interfaceInfo->findField(fieldNameAttr);
                            if (auto literalType = dyn_cast<mlir_ts::LiteralType>(fieldInfo->type))
                            {
                                if (literalType.getValue() == valueAttr)
                                {
                                    // enable safe cast found
                                    auto typeAliasNameUTF8 = MLIRHelper::getAnonymousName(loc_check(textRange), "ta_", getNamespaceName());
                                    auto typeAliasName = convertUTF8toWide(typeAliasNameUTF8);
                                    const_cast<GenContext &>(genContext)
                                        .typeAliasMap.insert({typeAliasNameUTF8, interfaceType});

                                    NodeFactory nf(NodeFactoryFlags::None);
                                    auto typeRef = nf.createTypeReferenceNode(nf.createIdentifier(typeAliasName));
                                    return addSafeCastStatement(objAccessExpression, typeRef, inverse, elseSafeCase, genContext);
                                }
                            }
                        }
                    }                    
                }
            }
        }

        return mlir::failure();
    }

    mlir::LogicalResult checkSafeCastPropertyAccess(Expression exprVal, Expression constVal,
                                                    bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        auto expr = stripParentheses(exprVal);
        if (expr == SyntaxKind::PropertyAccessExpression)
        {
            auto isConstVal = isConstValue(constVal, genContext);
            if (!isConstVal)
            {
                return mlir::failure();
            }

            auto propertyAccessExpressionOp = expr.as<PropertyAccessExpression>();
            auto objAccessExpression = propertyAccessExpressionOp->expression;
            auto typeOfObject = evaluate(objAccessExpression, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! SafeCastCheck: " << typeOfObject << "");

            auto val = mlirGen(constVal, genContext);
            return checkSafeCastPropertyAccessLogic(constVal, objAccessExpression, typeOfObject,
                                                    propertyAccessExpressionOp->name, val, inverse, elseSafeCase, genContext);
        }

        return mlir::failure();
    }

    mlir::LogicalResult checkSafeCastTypePredicate(Expression expr, mlir_ts::TypePredicateType typePredicateType, bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        return addSafeCastStatement(expr, typePredicateType.getElementType(), inverse, elseSafeCase, genContext);
    }

    mlir::LogicalResult checkSafeCast(Expression exprIn, mlir::Value conditionValue, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        auto expr = stripParentheses(exprIn);
        if (expr == SyntaxKind::CallExpression)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SafeCast: condition: " << conditionValue << "\n");

            if (auto callInd = conditionValue.getDefiningOp<mlir_ts::CallIndirectOp>())
            {
                auto funcType = callInd.getCallee().getType();

                auto resultType = mth.getReturnTypeFromFuncRef(funcType);

                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(resultType))
                {
                    // TODO: you need to find argument by using parameter name
                    auto callExpr = expr.as<CallExpression>();
                    if (typePredicateType.getParameterName().getValue() == THIS_NAME)
                    {
                        if (callExpr->expression == SyntaxKind::PropertyAccessExpression)
                        {
                            // in case of "this"
                            return checkSafeCastTypePredicate(
                                callExpr->expression.as<PropertyAccessExpression>()->expression, 
                                typePredicateType, 
                                false,
                                elseSafeCase,
                                genContext);                            
                        }
                    }
                    else if (typePredicateType.getParameterIndex() >= 0 && callExpr->arguments.size() > 0)
                    {
                        // in case of parameters
                        return checkSafeCastTypePredicate(
                            callExpr->arguments[typePredicateType.getParameterIndex()], 
                            typePredicateType,
                            false,
                            elseSafeCase, 
                            genContext);
                    }
                    else
                    {
                        llvm_unreachable("type predicate can't find parameter index. check funcProto context");
                    }
                }
            }

            return mlir::success();
        }
        else if (expr == SyntaxKind::PropertyAccessExpression)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SafeCast: condition: " << conditionValue << "\n");

            mlir_ts::TypePredicateType propertyType;
            if (auto loadOp = conditionValue.getDefiningOp<mlir_ts::LoadOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(loadOp.getType()))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto thisAccessor = conditionValue.getDefiningOp<mlir_ts::ThisAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(thisAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto thisIndirectAccessor = conditionValue.getDefiningOp<mlir_ts::ThisIndirectAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(thisIndirectAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto thisIndexAccessor = conditionValue.getDefiningOp<mlir_ts::ThisIndexAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(thisIndexAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto thisIndirectIndexAccessor = conditionValue.getDefiningOp<mlir_ts::ThisIndirectIndexAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(thisIndirectIndexAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto boundIndirectAccessor = conditionValue.getDefiningOp<mlir_ts::BoundIndirectAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(boundIndirectAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }
            else if (auto boundIndirectIndexAccessor = conditionValue.getDefiningOp<mlir_ts::BoundIndirectIndexAccessorOp>())
            {
                if (auto typePredicateType = dyn_cast<mlir_ts::TypePredicateType>(boundIndirectIndexAccessor.getType(0)))
                {
                    propertyType = typePredicateType;
                }
            }

            if (propertyType && propertyType.getParameterName().getValue() == THIS_NAME)
            {
                // in case of "this"
                return checkSafeCastTypePredicate(
                    expr.as<PropertyAccessExpression>()->expression, 
                    propertyType, 
                    false,
                    elseSafeCase,
                    genContext);                            
            }

            return mlir::success();
        }
        else if (expr == SyntaxKind::BinaryExpression)
        {
            auto binExpr = expr.as<BinaryExpression>();
            auto op = (SyntaxKind)binExpr->operatorToken;
            if (op == SyntaxKind::EqualsEqualsToken 
                || op == SyntaxKind::EqualsEqualsEqualsToken 
                || op == SyntaxKind::ExclamationEqualsToken 
                || op == SyntaxKind::ExclamationEqualsEqualsToken)
            {
                auto inverse = op == SyntaxKind::ExclamationEqualsToken || op == SyntaxKind::ExclamationEqualsEqualsToken;

                auto left = stripParentheses(binExpr->left);
                auto right = stripParentheses(binExpr->right);

                // TODO: refactor it
                // typeof
                if (mlir::failed(checkSafeCastTypeOf(left, right, inverse, elseSafeCase, genContext)))
                {
                    if (mlir::failed(checkSafeCastTypeOf(right, left, inverse, elseSafeCase, genContext)))
                    {
                        // property access
                        if (mlir::failed(checkSafeCastPropertyAccess(left, right, inverse, elseSafeCase, genContext)))
                        {
                            if (mlir::failed(checkSafeCastPropertyAccess(right, left, inverse, elseSafeCase, genContext)))
                            {
                                // undefined case
                                if (mlir::failed(checkSafeCastUndefined(left, right, !inverse, elseSafeCase, genContext)))
                                {
                                    return checkSafeCastUndefined(right, left, !inverse, elseSafeCase, genContext);
                                }
                            }
                        }
                    }
                }

                return mlir::success();
            }

            if (op == SyntaxKind::InstanceOfKeyword)
            {
                auto instanceOf = binExpr;
                if (instanceOf->left == SyntaxKind::Identifier)
                {
                    NodeFactory nf(NodeFactoryFlags::None);
                    return addSafeCastStatement(instanceOf->left, nf.createTypeReferenceNode(instanceOf->right),
                                                false, elseSafeCase, genContext);
                }
            }
        }
        else if (expr == SyntaxKind::PrefixUnaryExpression)
        {
            auto prefixExpr = expr.as<PrefixUnaryExpression>();
            auto opCode = prefixExpr->_operator;
            if (opCode == SyntaxKind::ExclamationToken)
            {
                auto expression = prefixExpr->operand;
                return checkSafeCastBoolean(expression, true, elseSafeCase, genContext);
            }
        }        
        else if (expr == SyntaxKind::Identifier)
        {
            // in case of boolean value
            return checkSafeCastBoolean(expr, false, elseSafeCase, genContext);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
    {
        auto location = loc(ifStatementAST);

        auto hasElse = !!ifStatementAST->elseStatement;

        // condition
        auto result = mlirGen(ifStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto condValue = V(result);

        // special case: in case of LiteralValue do not process If value is False
        std::optional<bool> literalValue;
        if (auto litType = mlir::dyn_cast<mlir_ts::LiteralType>(condValue.getType()))
        {
            if (auto boolVal = mlir::dyn_cast<mlir::BoolAttr>(litType.getValue()))
            {
                literalValue = boolVal.getValue();
            }
        }

        // default implementation of IfOp
        if (condValue.getType() != getBooleanType())
        {
            CAST(condValue, location, getBooleanType(), condValue, genContext);
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

        ElseSafeCase elseSafeCase{};
        {
            // check if we do safe-cast here
            SymbolTableScopeT varScope(symbolTable);
            checkSafeCast(ifStatementAST->expression, V(result), hasElse ? &elseSafeCase : nullptr, genContext);

            auto processIf = !literalValue.has_value() || literalValue.value();
            if (processIf)
            {
                auto result = mlirGen(ifStatementAST->thenStatement, genContext);
                EXIT_IF_FAILED(result)
            }
        }

        if (hasElse)
        {
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            SymbolTableScopeT varScope(symbolTable);
            if (elseSafeCase.safeType)
            {
                // add case statement
                addSafeCastStatement(elseSafeCase.expr, elseSafeCase.safeType, false, nullptr, genContext);
            }

            auto processIf = !literalValue.has_value() || !literalValue.value();
            if (processIf)
            {
                auto result = mlirGen(ifStatementAST->elseStatement, genContext);
                EXIT_IF_FAILED(result)
            }
        }

        builder.setInsertionPointAfter(ifOp);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(DoStatement doStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(doStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto doWhileOp = builder.create<mlir_ts::DoWhileOp>(location, types, operands);
        if (!label.empty())
        {
            doWhileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        const_cast<GenContext &>(genContext).isLoop = true;
        const_cast<GenContext &>(genContext).loopLabel = label;

        /*auto *cond =*/builder.createBlock(&doWhileOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&doWhileOp.getBody(), {}, types);

        // body in condition
        builder.setInsertionPointToStart(&doWhileOp.getBody().front());
        auto result2 = mlirGen(doStatementAST->statement, genContext);
        EXIT_IF_FAILED(result2)
        // just simple return, as body in cond
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointToStart(&doWhileOp.getCond().front());
        auto result = mlirGen(doStatementAST->expression, genContext);
        EXIT_IF_FAILED(result)
        auto conditionValue = V(result);

        if (conditionValue.getType() != getBooleanType())
        {
            CAST(conditionValue, location, getBooleanType(), conditionValue, genContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        builder.setInsertionPointAfter(doWhileOp);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(WhileStatement whileStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(whileStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto whileOp = builder.create<mlir_ts::WhileOp>(location, types, operands);
        if (!label.empty())
        {
            whileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        const_cast<GenContext &>(genContext).isLoop = true;
        const_cast<GenContext &>(genContext).loopLabel = label;

        /*auto *cond =*/builder.createBlock(&whileOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&whileOp.getBody(), {}, types);

        // condition
        builder.setInsertionPointToStart(&whileOp.getCond().front());
        auto result = mlirGen(whileStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto conditionValue = V(result);

        if (conditionValue.getType() != getBooleanType())
        {
            CAST(conditionValue, location, getBooleanType(), conditionValue, genContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        // body
        builder.setInsertionPointToStart(&whileOp.getBody().front());
        auto result2 = mlirGen(whileStatementAST->statement, genContext);
        EXIT_IF_FAILED(result2)
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(whileOp);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForStatement forStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forStatementAST);

        auto hasAwait = InternalFlags::ForAwait == (forStatementAST->internalFlags & InternalFlags::ForAwait);

        // initializer
        // TODO: why do we have ForInitialier
        if (isExpression(forStatementAST->initializer))
        {
            auto result = mlirGen(forStatementAST->initializer.as<Expression>(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto init = V(result);
            if (!init)
            {
                return mlir::failure();
            }
        }
        else if (forStatementAST->initializer == SyntaxKind::VariableDeclarationList)
        {
            auto result = mlirGen(forStatementAST->initializer.as<VariableDeclarationList>(), genContext);
            EXIT_IF_FAILED(result)
            if (failed(result))
            {
                return result;
            }
        }

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        mlir::Value asyncGroupResult;
        if (hasAwait)
        {
            auto groupType = mlir::async::GroupType::get(builder.getContext());
            auto blockSize = builder.create<mlir_ts::ConstantOp>(location, builder.getIndexAttr(0));
            auto asyncGroupOp = builder.create<mlir::async::CreateGroupOp>(location, groupType, blockSize);
            asyncGroupResult = asyncGroupOp.getResult();
            // operands.push_back(asyncGroupOp);
            // types.push_back(groupType);
        }

        auto forOp = builder.create<mlir_ts::ForOp>(location, types, operands);
        if (!label.empty())
        {
            forOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        const_cast<GenContext &>(genContext).isLoop = true;
        const_cast<GenContext &>(genContext).loopLabel = label;

        /*auto *cond =*/builder.createBlock(&forOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&forOp.getBody(), {}, types);
        /*auto *incr =*/builder.createBlock(&forOp.getIncr(), {}, types);

        builder.setInsertionPointToStart(&forOp.getCond().front());
        auto result = mlirGen(forStatementAST->condition, genContext);
        EXIT_IF_FAILED(result)
        auto conditionValue = V(result);
        if (conditionValue)
        {
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});
        }
        else
        {
            builder.create<mlir_ts::NoConditionOp>(location, mlir::ValueRange{});
        }

        // body
        builder.setInsertionPointToStart(&forOp.getBody().front());
        if (hasAwait)
        {
            if (forStatementAST->statement == SyntaxKind::Block)
            {
                auto firstStatement = forStatementAST->statement.as<ts::Block>()->statements.front();
                auto result = mlirGen(firstStatement, genContext);
                EXIT_IF_FAILED(result)
            }

            // TODO: we need to strip metadata to fix issue with debug info
            // async body
            auto isFailed = false;
            auto asyncExecOp = builder.create<mlir::async::ExecuteOp>(
                stripMetadata(location), mlir::TypeRange{}, mlir::ValueRange{}, mlir::ValueRange{},
                [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                    GenContext execOpBodyGenContext(genContext);
                    DITableScopeT debugAsyncCodeScope(debugScope);
                    MLIRDebugInfoHelper mdi(builder, debugScope);
                    
                    // TODO: temp hack to break wrong chain on scopes because 'await' create extra function wrap
                    mdi.clearDebugScope();
                    mdi.setLexicalBlock(location);

                    if (forStatementAST->statement == SyntaxKind::Block)
                    {
                        if (mlir::failed(mlirGen(forStatementAST->statement.as<ts::Block>(), execOpBodyGenContext, 1)))
                        {
                            isFailed = true;
                        }
                    }
                    else
                    {
                        if (mlir::failed(mlirGen(forStatementAST->statement, execOpBodyGenContext))) 
                        {
                            isFailed = true;
                        }
                    }

                    builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                });    

            if (isFailed)
            {
                return mlir::failure();
            }

            // add to group
            auto rankType = mlir::IndexType::get(builder.getContext());
            // TODO: should i replace with value from arg0?
            builder.create<mlir::async::AddToGroupOp>(location, rankType, asyncExecOp.getToken(), asyncGroupResult);
        }
        else
        {
            // default
            auto result = mlirGen(forStatementAST->statement, genContext);
            EXIT_IF_FAILED(result)
        }

        builder.create<mlir_ts::ResultOp>(location);

        // increment
        builder.setInsertionPointToStart(&forOp.getIncr().front());
        mlirGen(forStatementAST->incrementor, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(forOp);

        if (hasAwait)
        {
            // Not helping
            /*
            // async await all, see convert-to-llvm.mlir
            auto asyncExecAwaitAllOp =
                builder.create<mlir::async::ExecuteOp>(location, mlir::TypeRange{}, mlir::ValueRange{},
            mlir::ValueRange{},
                                                       [&](mlir::OpBuilder &builder, mlir::Location location,
            mlir::ValueRange values) { builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
                                                           builder.create<mlir::async::YieldOp>(location,
            mlir::ValueRange{});
                                                       });
            */

            // Wait for the completion of all subtasks.
            builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForInStatement forInStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forInStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S(".i"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S(".a"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forInStatementAST->expression);
        arrayVar->internalFlags |= InternalFlags::ForceConstRef;
        declarations.push_back(arrayVar);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        // condition
        // auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
        // nf.createCallExpression(nf.createIdentifier(S("#_last_field")), undefined, NodeArray<Expression>(_a)));
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S(LENGTH_FIELD_NAME))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forInStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = _i;

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forInStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGenES3(ForOfStatement forOfStatementAST, mlir::Value exprValue,
                                   const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(EXPR_TEMPVAR_NAME, exprValue.getType(), location);
        // somehow it is detected as external var, seems because it is contains external ref
        varDecl->setIgnoreCapturing();
        DECLARE(varDecl, exprValue);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S(".i"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S(".a"));
        auto arrayVar =
            nf.createVariableDeclaration(_a, undefined, undefined, nf.createIdentifier(S(EXPR_TEMPVAR_NAME)));
        arrayVar->internalFlags |= InternalFlags::ForceConstRef;

        declarations.push_back(arrayVar);

        // condition
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S(LENGTH_FIELD_NAME))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        NodeArray<VariableDeclaration> varOfConstDeclarations;
        auto _ci = nf.createIdentifier(S(".ci"));
        varOfConstDeclarations.push_back(nf.createVariableDeclaration(_ci, undefined, undefined, _i));
        auto varsOfConst = nf.createVariableDeclarationList(varOfConstDeclarations, NodeFlags::Const);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let /*varDeclList->flags*/);

        // in async exec, we will put first statement outside fo async.exec, to convert ref<int> into <int>
        statements.push_back(nf.createVariableStatement(undefined, varsOfConst));

        if (forOfStatementAST->initializer == SyntaxKind::VariableDeclarationList)
        {
            auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
            if (!varDeclList->declarations.empty())
            {
                varDeclList->declarations.front()->initializer = nf.createElementAccessExpression(_a, _ci);
                statements.push_back(nf.createVariableStatement(undefined, varDeclList));
            }
        }
        else
        {
            // set value
            statements.push_back(nf.createExpressionStatement(
                nf.createBinaryExpression(forOfStatementAST->initializer, nf.createToken(SyntaxKind::EqualsToken), nf.createElementAccessExpression(_a, _ci))
            ));
        }

        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);
        if (forOfStatementAST->awaitModifier)
        {
            forStatNode->internalFlags |= InternalFlags::ForAwait;
        }

        LLVM_DEBUG(printDebug(forStatNode););

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGenES2015(ForOfStatement forOfStatementAST, mlir::Value exprValue,
                                      const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(EXPR_TEMPVAR_NAME, exprValue.getType(), location);
        // somehow it is detected as external var, seems because it is contains external ref
        varDecl->setIgnoreCapturing();
        DECLARE(varDecl, exprValue);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _b = nf.createIdentifier(S(".b"));
        auto _next = nf.createIdentifier(S(ITERATOR_NEXT));
        auto _bVar = nf.createVariableDeclaration(_b, undefined, undefined, nf.createIdentifier(S(EXPR_TEMPVAR_NAME)));
        declarations.push_back(_bVar);

        NodeArray<Expression> nextArgs;

        auto _c = nf.createIdentifier(S(".c"));
        auto _done = nf.createIdentifier(S("done"));
        auto _value = nf.createIdentifier(S("value"));
        auto _cVar = nf.createVariableDeclaration(
            _c, undefined, undefined,
            nf.createCallExpression(nf.createPropertyAccessExpression(_b, _next), undefined, nextArgs));
        declarations.push_back(_cVar);

        // condition
        auto cond = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::ExclamationToken),
                                                   nf.createPropertyAccessExpression(_c, _done));

        // incr
        auto incr = nf.createBinaryExpression(
            _c, nf.createToken(SyntaxKind::EqualsToken),
            nf.createCallExpression(nf.createPropertyAccessExpression(_b, _next), undefined, nextArgs));

        // block
        NodeArray<ts::Statement> statements;

        if (forOfStatementAST->initializer == SyntaxKind::VariableDeclarationList)
        {
            auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
            if (!varDeclList->declarations.empty())
            {
                varDeclList->declarations.front()->initializer = nf.createPropertyAccessExpression(_c, _value);
                statements.push_back(nf.createVariableStatement(undefined, varDeclList));
            }
        }
        else
        {
            // set value
            statements.push_back(nf.createExpressionStatement(
                nf.createBinaryExpression(forOfStatementAST->initializer, nf.createToken(SyntaxKind::EqualsToken), nf.createPropertyAccessExpression(_c, _value))
            ));            
        }

        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let /*varDeclList->flags*/);
        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);
        if (forOfStatementAST->awaitModifier)
        {
            forStatNode->internalFlags |= InternalFlags::ForAwait;
        }

        LLVM_DEBUG(printDebug(forStatNode););

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext)
    {
        auto location = loc(forOfStatementAST);

        auto result = mlirGen(forOfStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        auto skip = isa<mlir_ts::ArrayType>(exprValue.getType()) 
                 || isa<mlir_ts::StringType>(exprValue.getType());
        // we need to ignore SYMBOL_ITERATOR for array to use simplier method and do not cause the stackoverflow
        if (!skip)
        {
            auto iteratorIdent = (forOfStatementAST->awaitModifier) ? SYMBOL_ASYNC_ITERATOR : SYMBOL_ITERATOR;
            if (auto iteratorType = evaluateProperty(location, exprValue, iteratorIdent, genContext))
            {
                if (auto iteratorValue = mlirGenCallThisMethod(location, exprValue, iteratorIdent, undefined, undefined, genContext))
                {
                    exprValue = V(iteratorValue);
                }
            }

            auto propertyType = evaluateProperty(location, exprValue, ITERATOR_NEXT, genContext);
            if (propertyType)
            {
                if (mlir::succeeded(mlirGenES2015(forOfStatementAST, exprValue, genContext)))
                {
                    return mlir::success();
                }
            }
        }

        return mlirGenES3(forOfStatementAST, exprValue, genContext);
    }

    mlir::LogicalResult mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext)
    {
        auto location = loc(labeledStatementAST);

        label = MLIRHelper::getName(labeledStatementAST->label);

        auto kind = (SyntaxKind)labeledStatementAST->statement;
        if (kind == SyntaxKind::EmptyStatement && StringRef(label).starts_with(GENERATOR_STATELABELPREFIX))
        {
            builder.create<mlir_ts::StateLabelOp>(location, builder.getStringAttr(label));
            return mlir::success();
        }

        auto noLabelOp = kind == SyntaxKind::WhileStatement || kind == SyntaxKind::DoStatement ||
                         kind == SyntaxKind::ForStatement || kind == SyntaxKind::ForInStatement ||
                         kind == SyntaxKind::ForOfStatement;

        if (noLabelOp)
        {
            return mlirGen(labeledStatementAST->statement, genContext);
        }

        auto labelOp = builder.create<mlir_ts::LabelOp>(location, builder.getStringAttr(label));

        // add merge block
        labelOp.addMergeBlock();
        auto *mergeBlock = labelOp.getMergeBlock();

        builder.setInsertionPointToStart(mergeBlock);

        auto res = mlirGen(labeledStatementAST->statement, genContext);

        builder.setInsertionPointAfter(labelOp);

        return res;
    }

    mlir::LogicalResult mlirGen(DebuggerStatement debuggerStatementAST, const GenContext &genContext)
    {
        auto location = loc(debuggerStatementAST);

        builder.create<mlir_ts::DebuggerOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext)
    {
        auto location = loc(continueStatementAST);

        auto label = MLIRHelper::getName(continueStatementAST->label);

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::LoopScope, label, &genContext));

        builder.create<mlir_ts::ContinueOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(BreakStatement breakStatementAST, const GenContext &genContext)
    {
        auto location = loc(breakStatementAST);

        auto label = MLIRHelper::getName(breakStatementAST->label);

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::LoopScope, label, &genContext));

        builder.create<mlir_ts::BreakOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGenSwitchCase(mlir::Location location, Expression switchExpr, mlir::Value switchValue,
                                          NodeArray<ts::CaseOrDefaultClause> &clauses, int index,
                                          mlir::Block *mergeBlock, mlir::Block *&defaultBlock,
                                          SmallVector<mlir::cf::CondBranchOp> &pendingConditions,
                                          SmallVector<mlir::cf::BranchOp> &pendingBranches,
                                          mlir::Operation *&previousConditionOrFirstBranchOp,
                                          std::function<void(Expression, mlir::Value)> extraCode,
                                          const GenContext &genContext)
    {
        SymbolTableScopeT safeCastVarScope(symbolTable);

        enum
        {
            trueIndex = 0,
            falseIndex = 1
        };

        auto caseBlock = clauses[index];
        auto statements = caseBlock->statements;
        // inline block
        // TODO: should I inline block as it is isolator of local vars?
        if (statements.size() == 1)
        {
            auto firstStatement = statements.front();
            if ((SyntaxKind)firstStatement == SyntaxKind::Block)
            {
                statements = statements.front().as<ts::Block>()->statements;
            }
        }

        auto setPreviousCondOrJumpOp = [&](mlir::Operation *jump, mlir::Block *where) {
            if (auto condOp = dyn_cast<mlir::cf::CondBranchOp>(jump))
            {
                condOp->setSuccessor(where, falseIndex);
                return;
            }

            if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(jump))
            {
                branchOp.setDest(where);
                return;
            }

            llvm_unreachable("not implemented");
        };

        // condition
        auto isDefaultCase = SyntaxKind::DefaultClause == (SyntaxKind)caseBlock;
        auto isDefaultAsFirstCase = index == 0 && clauses.size() > 1;
        if (SyntaxKind::CaseClause == (SyntaxKind)caseBlock)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto caseConditionBlock = builder.createBlock(mergeBlock);
            if (previousConditionOrFirstBranchOp)
            {
                setPreviousCondOrJumpOp(previousConditionOrFirstBranchOp, caseConditionBlock);
            }

            auto caseExpr = caseBlock.as<CaseClause>()->expression;
            auto result = mlirGen(caseExpr, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto caseValue = V(result);

            extraCode(caseExpr, caseValue);

            auto switchValueEffective = switchValue;
            auto actualCaseType = mth.stripLiteralType(caseValue.getType());
            if (switchValue.getType() != actualCaseType)
            {
                CAST(switchValueEffective, location, actualCaseType, switchValue, genContext);
            }

            auto condition = builder.create<mlir_ts::LogicalBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken),
                switchValueEffective, caseValue);

            CAST_A(conditionI1, location, builder.getI1Type(), condition, genContext);

            auto condBranchOp = builder.create<mlir::cf::CondBranchOp>(location, conditionI1, mergeBlock,
                                                                   /*trueArguments=*/mlir::ValueRange{},
                                                                   defaultBlock ? defaultBlock : mergeBlock,
                                                                   /*falseArguments=*/mlir::ValueRange{});

            previousConditionOrFirstBranchOp = condBranchOp;

            pendingConditions.push_back(condBranchOp);
        }
        else if (isDefaultAsFirstCase)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            /*auto defaultCaseJumpBlock =*/builder.createBlock(mergeBlock);

            // this is first default and there is more conditions
            // add jump to first condition
            auto branchOp = builder.create<mlir::cf::BranchOp>(location, mergeBlock);

            previousConditionOrFirstBranchOp = branchOp;
        }

        // statements block
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto caseBodyBlock = builder.createBlock(mergeBlock);
            if (isDefaultCase)
            {
                defaultBlock = caseBodyBlock;
                if (!isDefaultAsFirstCase && previousConditionOrFirstBranchOp)
                {
                    setPreviousCondOrJumpOp(previousConditionOrFirstBranchOp, caseBodyBlock);
                }
            }

            // set pending BranchOps
            for (auto pendingBranch : pendingBranches)
            {
                pendingBranch.setDest(caseBodyBlock);
            }

            pendingBranches.clear();

            for (auto pendingCondition : pendingConditions)
            {
                pendingCondition.setSuccessor(caseBodyBlock, trueIndex);
            }

            pendingConditions.clear();

            // process body case
            if (genContext.generatedStatements.size() > 0)
            {
                // auto generated code
                for (auto &statement : genContext.generatedStatements)
                {
                    if (failed(mlirGen(statement, genContext)))
                    {
                        return mlir::failure();
                    }
                }

                // clean up
                const_cast<GenContext &>(genContext).generatedStatements.clear();
            }

            auto hasBreak = false;
            for (auto statement : statements)
            {
                if ((SyntaxKind)statement == SyntaxKind::BreakStatement)
                {
                    hasBreak = true;
                    break;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            // exit;
            auto branchOp = builder.create<mlir::cf::BranchOp>(location, mergeBlock);
            if (!hasBreak && !isDefaultCase)
            {
                pendingBranches.push_back(branchOp);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(switchStatementAST);

        auto switchExpr = switchStatementAST->expression;
        auto result = mlirGen(switchExpr, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto switchValue = V(result);

        auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

        GenContext switchGenContext(genContext);
        switchGenContext.allocateVarsOutsideOfOperation = true;
        switchGenContext.currentOperation = switchOp;
        switchGenContext.insertIntoParentScope = true;

        // add merge block
        switchOp.addMergeBlock();
        auto *mergeBlock = switchOp.getMergeBlock();

        auto &clauses = switchStatementAST->caseBlock->clauses;

        SmallVector<mlir::cf::CondBranchOp> pendingConditions;
        SmallVector<mlir::cf::BranchOp> pendingBranches;
        mlir::Operation *previousConditionOrFirstBranchOp = nullptr;
        mlir::Block *defaultBlock = nullptr;

        // to support safe cast
        std::function<void(Expression, mlir::Value)> safeCastLogic;
        if (switchExpr == SyntaxKind::PropertyAccessExpression)
        {
            auto propertyAccessExpressionOp = switchExpr.as<PropertyAccessExpression>();
            auto objAccessExpression = propertyAccessExpressionOp->expression;
            auto typeOfObject = evaluate(objAccessExpression, switchGenContext);
            auto name = propertyAccessExpressionOp->name;

            safeCastLogic = [=, &switchGenContext](Expression caseExpr, mlir::Value constVal) {
                GenContext safeCastGenContext(switchGenContext);
                switchGenContext.insertIntoParentScope = false;

                // Safe Cast
                if (mlir::failed(checkSafeCastTypeOf(switchExpr, caseExpr, false, nullptr, switchGenContext)))
                {
                    checkSafeCastPropertyAccessLogic(caseExpr, objAccessExpression, typeOfObject, name, constVal,
                                                     false, nullptr, switchGenContext);
                }
            };
        }
        else
        {
            safeCastLogic = [&](Expression caseExpr, mlir::Value constVal) {};
        }

        // process without default
        for (int index = 0; index < clauses.size(); index++)
        {
            if (mlir::failed(mlirGenSwitchCase(location, switchExpr, switchValue, clauses, index, mergeBlock,
                                               defaultBlock, pendingConditions, pendingBranches,
                                               previousConditionOrFirstBranchOp, safeCastLogic, switchGenContext)))
            {
                return mlir::failure();
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! SWITCH: " << switchOp << "\n");

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
    {
        auto location = loc(throwStatementAST);

        auto result = mlirGen(throwStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exception = V(result);

        auto throwOp = builder.create<mlir_ts::ThrowOp>(location, exception);

        if (!genContext.allowPartialResolve)
        {
            MLIRRTTIHelperVC rtti(builder, theModule, compileOptions);
            if (!rtti.setRTTIForType(
                location, exception.getType(), 
                [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); }))
            {
                emitError(location, "Not supported type in throw");
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(TryStatement tryStatementAST, const GenContext &genContext)
    {
        auto location = loc(tryStatementAST);

        std::string varName;
        auto catchClause = tryStatementAST->catchClause;
        if (catchClause)
        {
            auto varDecl = catchClause->variableDeclaration;
            if (varDecl)
            {
                varName = MLIRHelper::getName(varDecl->name);
                if (mlir::failed(mlirGen(varDecl, VariableType::Let, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        const_cast<GenContext &>(genContext).funcOp.setPersonalityAttr(builder.getBoolAttr(true));

        auto tryOp = builder.create<mlir_ts::TryOp>(location);

        GenContext tryGenContext(genContext);
        // TODO: why do I need to allocate variables outside of "try" block?
        // well - short answer: to get access to vars in nested blocks for example 'cleanup'
        tryGenContext.allocateUsingVarsOutsideOfOperation = true;
        tryGenContext.currentOperation = tryOp;

        SmallVector<mlir::Type, 0> types;

        /*auto *body =*/builder.createBlock(&tryOp.getBody(), {}, types);
        /*auto cleanup =*/builder.createBlock(&tryOp.getCleanup(), {}, types);
        /*auto *catches =*/builder.createBlock(&tryOp.getCatches(), {}, types);
        /*auto *finallyBlock =*/builder.createBlock(&tryOp.getFinally(), {}, types);

        {
            // body
            builder.setInsertionPointToStart(&tryOp.getBody().front());

            // prepare custom scope
            SymbolTableScopeT varScope(symbolTable);
            GenContext tryBodyGenContext(tryGenContext);
            tryBodyGenContext.parentBlockContext = &tryGenContext;

            auto usingVars = std::make_unique<SmallVector<ts::VariableDeclarationDOM::TypePtr>>();
            tryBodyGenContext.usingVars = usingVars.get();

            auto result = mlirGenNoScopeVarsAndDisposable(tryStatementAST->tryBlock, tryBodyGenContext);
            EXIT_IF_FAILED(result)

            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScopeKeepAfterUse, {}, &tryBodyGenContext));

            // terminator
            builder.create<mlir_ts::ResultOp>(location);

            // cleanup
            builder.setInsertionPointToStart(&tryOp.getCleanup().front());
            // we need to call dispose for those which are in "using"
            // usingVars are empty here
            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScope, {}, &tryBodyGenContext));

            // terminator
            builder.create<mlir_ts::ResultOp>(location);
        }

        // catches
        builder.setInsertionPointToStart(&tryOp.getCatches().front());
        if (catchClause && catchClause->block)
        {
            auto location = loc(catchClause->block);
            if (!varName.empty())
            {
                MLIRCodeLogic mcl(builder);
                auto varInfo = resolveIdentifier(location, varName, tryGenContext);
                auto varRef = mcl.GetReferenceFromValue(location, varInfo);
                builder.create<mlir_ts::CatchOp>(location, varRef);

                if (!genContext.allowPartialResolve)
                {
                    MLIRRTTIHelperVC rtti(builder, theModule, compileOptions);
                    if (!rtti.setRTTIForType(
                        location, 
                        varInfo.getType(),
                        [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); }))
                    {
                        emitError(location, "Not supported type in catch");
                        return mlir::failure();
                    }
                }
            }

            auto result = mlirGen(tryStatementAST->catchClause->block, tryGenContext);
            EXIT_IF_FAILED(result)
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // finally
        builder.setInsertionPointToStart(&tryOp.getFinally().front());
        if (tryStatementAST->finallyBlock)
        {
            auto result = mlirGen(tryStatementAST->finallyBlock, tryGenContext);
            EXIT_IF_FAILED(result)
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(tryOp);
        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
    {
        return mlirGen(unaryExpressionAST.as<Expression>(), genContext);
    }

    ValueOrLogicalResult mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext)
    {
        return mlirGen(leftHandSideExpressionAST.as<Expression>(), genContext);
    }

    ValueOrLogicalResult mlirGenPrefixUnaryExpression(mlir::Location location, SyntaxKind opCode, mlir_ts::ConstantOp constantOp,
                                                      const GenContext &genContext)
    {
        mlir::Value value;
        auto valueAttr = constantOp.getValueAttr();

        switch (opCode)
        {
            case SyntaxKind::PlusToken:
                value = 
                    mlir::TypeSwitch<mlir::Attribute, mlir::Value>(valueAttr)
                        .Case<mlir::IntegerAttr>([&](auto intAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, constantOp.getType(), builder.getIntegerAttr(intAttr.getType(), intAttr.getValue()));
                        })
                        .Case<mlir::FloatAttr>([&](auto floatAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, constantOp.getType(), builder.getFloatAttr(floatAttr.getType(), floatAttr.getValue()));
                        })
                        .Case<mlir::StringAttr>([&](auto strAttr) {
#ifdef NUMBER_F64
                            auto floatType = mlir::Float64Type::get(builder.getContext());
#else
                            auto floatType = mlir::Float32Type::get(builder.getContext());
#endif                            
                            APFloat fValue(APFloatBase::IEEEdouble());
                            if (llvm::errorToBool(fValue.convertFromString(strAttr.getValue(), APFloat::rmNearestTiesToEven).takeError()))
                            {
                                fValue = APFloat::getNaN(fValue.getSemantics());
                            }

                            return V(builder.create<mlir_ts::ConstantOp>(
                                location, floatType, builder.getFloatAttr(floatType, fValue)));
                        })
                        .Default([](auto) {
                            return mlir::Value();
                        });                        
                break;
            case SyntaxKind::MinusToken:
                value = 
                    mlir::TypeSwitch<mlir::Attribute, mlir::Value>(valueAttr)
                        .Case<mlir::IntegerAttr>([&](auto intAttr) {
                            // TODO: convert unsiged int type into signed
                            auto intType = mlir::cast<mlir::IntegerType>(intAttr.getType());
                            auto constType = constantOp.getType();
                            auto valAttr = intAttr;
                            if (intType.isSignless())
                            {
                                intType = builder.getIntegerType(intType.getWidth(), true);
                                valAttr = builder.getIntegerAttr(intType, -intAttr.getValue());
                                constType = mlir_ts::LiteralType::get(valAttr, intType);
                            } 
                            else if (intType.isSigned())
                            {
                                valAttr = builder.getIntegerAttr(intType, -intAttr.getValue());
                                constType = mlir_ts::LiteralType::get(valAttr, intType);
                            }
                            else if (intType.getWidth() <= 32)
                            {
                                intType = builder.getIntegerType(intType.getWidth() * 2, true);
                                auto newVal = -(intAttr.getValue().zext(intType.getWidth()));
                                valAttr = builder.getIntegerAttr(intType, newVal);
                                constType = mlir_ts::LiteralType::get(valAttr, intType);
                            }
                            else
                            {
                                SmallVector<char> res;
                                intAttr.getValue().toString(res, 10, false);
                                emitError(location) << "can't apply '-'. Too big value: " << std::string(res.data(), res.size()) << "";
                                return mlir::Value();
                            }

                            return (mlir::Value) builder.create<mlir_ts::ConstantOp>(location, constType, valAttr);
                        })
                        .Case<mlir::FloatAttr>([&](auto floatAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, constantOp.getType(), builder.getFloatAttr(floatAttr.getType(), -floatAttr.getValue()));
                        })
                        .Case<mlir::StringAttr>([&](auto strAttr) {
#ifdef NUMBER_F64
                            auto floatType = mlir::Float64Type::get(builder.getContext());
#else                            
                            auto floatType = mlir::Float32Type::get(builder.getContext());
#endif
                            APFloat fValue(APFloatBase::IEEEdouble());                            
                            if (llvm::errorToBool(fValue.convertFromString(strAttr.getValue(), APFloat::rmNearestTiesToEven).takeError()))
                            {
                                fValue = APFloat::getNaN(fValue.getSemantics());
                            }

                            return V(builder.create<mlir_ts::ConstantOp>(
                                location, floatType, builder.getFloatAttr(floatType, -fValue)));
                        })                        
                        .Default([](auto) {
                            return mlir::Value();
                        });
                break;
            case SyntaxKind::TildeToken:
                // TODO: improvements required: use the same function to convert string into int as in LiteralNumeric
                // check if you can use it on 64 bits, check JS code for it
                value = 
                    mlir::TypeSwitch<mlir::Attribute, mlir::Value>(valueAttr)
                        .Case<mlir::IntegerAttr>([&](auto intAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, constantOp.getType(), builder.getIntegerAttr(intAttr.getType(), ~intAttr.getValue()));
                        })
                        .Case<mlir::StringAttr>([&](auto strAttr) {
                            auto intType = mlir::IntegerType::get(builder.getContext(), 32);
                            APInt iValue(32, 0);
                            if (!llvm::to_integer(strAttr.getValue(), iValue))
                            {
                                return mlir::Value();
                            }

                            return V(builder.create<mlir_ts::ConstantOp>(
                                location, intType, builder.getIntegerAttr(intType, ~iValue)));
                        })                         
                        .Default([](auto) {
                            return mlir::Value();
                        });
                break;
            case SyntaxKind::ExclamationToken:
                value = 
                    mlir::TypeSwitch<mlir::Attribute, mlir::Value>(valueAttr)
                        .Case<mlir::IntegerAttr>([&](auto intAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, getBooleanType(), builder.getBoolAttr(!(intAttr.getValue())));
                        })
                        .Case<mlir::StringAttr>([&](auto strAttr) {
                            return builder.create<mlir_ts::ConstantOp>(
                                location, getBooleanType(), builder.getBoolAttr(!(strAttr.getValue().empty())));
                        })                         
                        .Default([](auto) {
                            return mlir::Value();
                        });
                break;
            default:
                llvm_unreachable("not implemented");
        }

        return value;
    }

    ValueOrLogicalResult mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(prefixUnaryExpressionAST);

        auto opCode = prefixUnaryExpressionAST->_operator;

        auto expression = prefixUnaryExpressionAST->operand;
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        // special case "-" for literal value
        if (opCode == SyntaxKind::PlusToken || opCode == SyntaxKind::MinusToken || opCode == SyntaxKind::TildeToken || opCode == SyntaxKind::ExclamationToken)
        {
            if (auto constantOp = expressionValue.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto res = mlirGenPrefixUnaryExpression(location, opCode, constantOp, genContext);
                EXIT_IF_FAILED(res)
                if (res.value)
                {
                    return res.value;
                }
            }
        }

        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:
            {
                auto boolValue = expressionValue;
                if (expressionValue.getType() != getBooleanType())
                {
                    CAST(boolValue, location, getBooleanType(), expressionValue, genContext);
                }

                return V(builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(),
                                                                    builder.getI32IntegerAttr((int)opCode), boolValue));
            }
        case SyntaxKind::TildeToken:
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
            {
                auto numberValue = expressionValue;
                if (expressionValue.getType() != getNumberType() && !expressionValue.getType().isIntOrIndexOrFloat())
                {
                    CAST(numberValue, location, getNumberType(), expressionValue, genContext);
                }

                return V(builder.create<mlir_ts::ArithmeticUnaryOp>(
                    location, numberValue.getType(), builder.getI32IntegerAttr((int)opCode), numberValue));
            }
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return V(builder.create<mlir_ts::PrefixUnaryOp>(location, expressionValue.getType(),
                                                            builder.getI32IntegerAttr((int)opCode), expressionValue));
        default:
            llvm_unreachable("not implemented");
        }
    }

    ValueOrLogicalResult mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(postfixUnaryExpressionAST);

        auto opCode = postfixUnaryExpressionAST->_operator;

        auto expression = postfixUnaryExpressionAST->operand;
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        switch (opCode)
        {
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return V(builder.create<mlir_ts::PostfixUnaryOp>(location, expressionValue.getType(),
                                                             builder.getI32IntegerAttr((int)opCode), expressionValue));
        default:
            llvm_unreachable("not implemented");
        }
    }

    // TODO: rewrite code, you can set IfOp result type later, see function anyOrUndefined
    ValueOrLogicalResult mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext)
    {
        auto location = loc(conditionalExpressionAST);

        // condition
        auto condExpression = conditionalExpressionAST->condition;
        auto result = mlirGen(condExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)

        auto condValue = V(result);
        if (condValue.getType() != getBooleanType())
        {
            CAST(condValue, location, getBooleanType(), condValue, genContext);
        }

        // detect value type
        // TODO: sync types for 'when' and 'else'

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{getVoidType()}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto whenTrueExpression = conditionalExpressionAST->whenTrue;

        ElseSafeCase elseSafeCase;
        mlir::Value resultTrue;
        {
            // check if we do safe-cast here
            SymbolTableScopeT varScope(symbolTable);
            checkSafeCast(conditionalExpressionAST->condition, V(result), &elseSafeCase, genContext);
            auto result = mlirGen(whenTrueExpression, genContext);
            if (!genContext.allowPartialResolve)
            {
                EXIT_IF_FAILED_OR_NO_VALUE(result)
            }
            
            resultTrue = V(result);
        }

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto whenFalseExpression = conditionalExpressionAST->whenFalse;

        mlir::Value resultFalse;
        {
            SymbolTableScopeT varScope(symbolTable);
            if (elseSafeCase.safeType)
            {
                addSafeCastStatement(elseSafeCase.expr, elseSafeCase.safeType, false, nullptr, genContext);
            }        
            
            auto result2 = mlirGen(whenFalseExpression, genContext);
            if (!genContext.allowPartialResolve)
            {
                EXIT_IF_FAILED_OR_NO_VALUE(result2)
            }

            resultFalse = V(result2);
        }

        if (resultTrue && resultFalse)
        {
            auto defaultUnionType = getUnionType(location, resultTrue.getType(), resultFalse.getType());
            auto merged = false;
            auto resultType = mth.findBaseType(resultTrue.getType(), resultFalse.getType(), merged, defaultUnionType);

            ifOp.getResult(0).setType(resultType);

            CAST_A(falseRes, location, resultType, resultFalse, genContext)
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{falseRes});

            // finish type of IfOp and WhenTrue clause
            builder.setInsertionPointToEnd(&ifOp.getThenRegion().back());

            CAST_A(trueRes, location, resultType, resultTrue, genContext);
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{trueRes});
        }
        else
        {
            // to support partial result
            auto partialResult = resultTrue ? resultTrue : resultFalse;
            if (partialResult)
            {
                ifOp.getResult(0).setType(partialResult.getType());
            }
            else
            {
                return mlir::failure();
            }
        }

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    }

    ValueOrLogicalResult mlirGenAndOrLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext,
                                           bool andOp, bool saveResult)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        // condition
        auto result = mlirGen(leftExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto leftExpressionValue = V(result);

        CAST_A(condValue, location, getBooleanType(), leftExpressionValue, genContext);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{leftExpressionValue.getType()}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

        ElseSafeCase elseSafeCase;
        mlir::Value resultTrue;
        {
            if (andOp)
            {
                // check if we do safe-cast here
                SymbolTableScopeT varScope(symbolTable);
                checkSafeCast(leftExpression, V(result), &elseSafeCase, genContext);

                auto result = mlirGen(rightExpression, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                resultTrue = V(result);
            }
            else
            {
                resultTrue = leftExpressionValue;
                if (auto optType = dyn_cast<mlir_ts::OptionalType>(resultTrue.getType()))
                {
                    resultTrue = builder.create<mlir_ts::CastOp>(location, optType.getElementType(), resultTrue);
                }
            }

            if (andOp)
            {
                VALIDATE(resultTrue, location)
            }
        }

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        mlir::Value resultFalse;
        {        
            if (andOp)
            {
                resultFalse = leftExpressionValue;
                if (auto optType = dyn_cast<mlir_ts::OptionalType>(resultFalse.getType()))
                {
                    resultFalse = builder.create<mlir_ts::CastOp>(location, optType.getElementType(), resultFalse);
                }
            }
            else
            {
                SymbolTableScopeT varScope(symbolTable);
                if (elseSafeCase.safeType)
                {
                    addSafeCastStatement(elseSafeCase.expr, elseSafeCase.safeType, false, nullptr, genContext);
                }   

                auto result = mlirGen(rightExpression, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                resultFalse = V(result);
            }

            if (!andOp)
            {
                VALIDATE(resultFalse, location)
            }
        }

        auto resultType = getUnionType(location, resultTrue.getType(), resultFalse.getType());

        ifOp->getResult(0).setType(resultType);

        // sync right part
        if (resultType != resultFalse.getType())
        {
            CAST(resultFalse, location, resultType, resultFalse, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointToEnd(&ifOp.getThenRegion().back());

        // sync left part
        if (resultType != resultTrue.getType())
        {
            CAST(resultTrue, location, resultType, resultTrue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        // end of setting result for left part

        builder.setInsertionPointAfter(ifOp);

        auto resultFirst = ifOp.getResults().front();
        if (saveResult)
        {
            return mlirGenSaveLogicOneItem(location, leftExpressionValue, resultFirst, genContext);
        }

        return resultFirst;
    }

    ValueOrLogicalResult mlirGenQuestionQuestionLogic(BinaryExpression binaryExpressionAST, bool saveResult,
                                                      const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        // condition
        auto result = mlirGen(leftExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto leftExpressionValue = V(result);

        auto resultWhenFalseType = evaluate(rightExpression, genContext);
        auto defaultUnionType = getUnionType(location, leftExpressionValue.getType(), resultWhenFalseType);
        auto merged = false;
        auto resultType = mth.findBaseType(resultWhenFalseType, leftExpressionValue.getType(), merged, defaultUnionType);

        // extarct value from optional type
        auto actualLeftValue = leftExpressionValue;
        auto hasOptional = false;
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(actualLeftValue.getType()))
        {
            hasOptional = true;
            CAST(actualLeftValue, location, optType.getElementType(), leftExpressionValue, genContext);
        }

        CAST_A(opaqueValueOfLeftValue, location, getOpaqueType(), actualLeftValue, genContext);

        auto nullVal = builder.create<mlir_ts::NullOp>(location, getNullType());

        auto compareToNull = builder.create<mlir_ts::LogicalBinaryOp>(
            location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsEqualsToken), opaqueValueOfLeftValue,
            nullVal);

        mlir::Value ifCond = compareToNull;
        if (hasOptional)
        {
            CAST_A(hasValue, location, getBooleanType(), leftExpressionValue, genContext);      
            CAST_A(isFalse, location, getBooleanType(), mlirGenBooleanValue(location, false), genContext);
            auto compareToFalse = builder.create<mlir_ts::LogicalBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsEqualsToken), isFalse,
                hasValue);

            auto orOp = builder.create<mlir_ts::ArithmeticBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::BarToken), compareToFalse,
                compareToNull);   

            ifCond = orOp;            
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, ifCond, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto result2 = mlirGen(rightExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result2)
        auto resultTrue = V(result2);

        // sync left part
        if (resultType != resultTrue.getType())
        {
            CAST(resultTrue, location, resultType, resultTrue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto resultFalse = leftExpressionValue;

        // sync right part
        if (resultType != resultFalse.getType())
        {
            CAST(resultFalse, location, resultType, resultFalse, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        auto ifResult = ifOp.getResults().front();
        if (saveResult)
        {
            return mlirGenSaveLogicOneItem(location, leftExpressionValue, ifResult, genContext);
        }

        return ifResult;
    }

    ValueOrLogicalResult mlirGenInLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        // Supports only array now
        auto location = loc(binaryExpressionAST);

        NodeFactory nf(NodeFactoryFlags::None);

        if (auto hasLength = evaluateProperty(binaryExpressionAST->right, LENGTH_FIELD_NAME, genContext))
        {
            auto cond1 = nf.createBinaryExpression(
                binaryExpressionAST->left, nf.createToken(SyntaxKind::LessThanToken),
                nf.createPropertyAccessExpression(binaryExpressionAST->right, nf.createIdentifier(S(LENGTH_FIELD_NAME))));

            auto cond2 = nf.createBinaryExpression(
                binaryExpressionAST->left, nf.createToken(SyntaxKind::GreaterThanEqualsToken), nf.createNumericLiteral(S("0")));

            auto cond = nf.createBinaryExpression(cond1, nf.createToken(SyntaxKind::AmpersandAmpersandToken), cond2);

            return mlirGen(cond, genContext);
        }

        auto resultLeft = mlirGen(binaryExpressionAST->left, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(resultLeft)
        auto leftExpressionValue = V(resultLeft);

        if (!isConstValue(leftExpressionValue))
        {
            emitError(loc(binaryExpressionAST->left), "not supported");
            return mlir::failure();
        }

        auto resultRight = mlirGen(binaryExpressionAST->right, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(resultRight)
        auto rightExpressionValue = V(resultRight);

        if (isa<mlir_ts::UnionType>(rightExpressionValue.getType()))
        {
            emitError(loc(binaryExpressionAST->right), "not supported");
            return mlir::failure();
        }

        if (auto constantOp = leftExpressionValue.getDefiningOp<mlir_ts::ConstantOp>())        
        {
            auto hasField = !!mth.getFieldTypeByFieldName(rightExpressionValue.getType(), constantOp.getValue());
            return mlirGenBooleanValue(loc(binaryExpressionAST->right), hasField);
        }

        emitError(location, "not supported");
        return mlir::failure();
    }

    ValueOrLogicalResult mlirGenCallThisMethod(mlir::Location location, mlir::Value thisValue, StringRef methodName,
                                               NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments,
                                               const GenContext &genContext)
    {
        // to remove temp var after call
        SymbolTableScopeT varScope(symbolTable);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_TEMPVAR_NAME, thisValue.getType(), location);
        DECLARE(varDecl, thisValue);

        NodeFactory nf(NodeFactoryFlags::None);

        auto thisToken = nf.createIdentifier(S(THIS_TEMPVAR_NAME));
        auto callLogic = nf.createCallExpression(
            nf.createPropertyAccessExpression(thisToken, nf.createIdentifier(stows(methodName.str()))), typeArguments,
            arguments);

        return mlirGen(callLogic, genContext);
    }

    mlir::Value mlirGenInstanceOfOpaque(mlir::Location location, mlir::Value thisPtrValue, mlir::Value classRefVal, const GenContext &genContext)
    {
        // get VTable we can use VTableOffset
        auto vtablePtr = builder.create<mlir_ts::VTableOffsetRefOp>(location, getOpaqueType(),
                                                                    thisPtrValue, 0 /*VTABLE index*/);

        // get InstanceOf method, this is 0 index in vtable
        auto instanceOfPtr = builder.create<mlir_ts::VTableOffsetRefOp>(
            location, getOpaqueType(), vtablePtr, 0 /*InstanceOf index*/);

        if (auto classType = dyn_cast<mlir_ts::ClassType>(classRefVal.getType()))
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());

            auto resultRtti = mlirGenPropertyAccessExpression(location, classRefVal, RTTI_NAME, genContext);
            if (!resultRtti)
            {
                return mlir::Value();
            }

            auto rttiOfClassValue = V(resultRtti);
            if (classInfo->isDynamicImport)
            {
                if (auto valueRefType = dyn_cast<mlir_ts::RefType>(rttiOfClassValue.getType()))
                {
                    rttiOfClassValue = builder.create<mlir_ts::LoadOp>(location, valueRefType.getElementType(), rttiOfClassValue);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }

            assert(rttiOfClassValue);

            auto instanceOfFuncType = mlir_ts::FunctionType::get(
                builder.getContext(), SmallVector<mlir::Type>{getOpaqueType(), getStringType()},
                SmallVector<mlir::Type>{getBooleanType()});

            // TODO: check result
            auto result = cast(location, instanceOfFuncType, instanceOfPtr, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto funcPtr = V(result);

            // call methos, we need to send, this, and rtti info
            auto callResult = builder.create<mlir_ts::CallIndirectOp>(
                MLIRHelper::getCallSiteLocation(funcPtr, location),
                funcPtr, mlir::ValueRange{thisPtrValue, rttiOfClassValue});

            return callResult.getResult(0);
        }

        // error
        return mlir::Value();
    }    

    ValueOrLogicalResult mlirGenInstanceOfLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        // check if we need to call hasInstance
        if (auto hasInstanceType = evaluateProperty(binaryExpressionAST->right, SYMBOL_HAS_INSTANCE, genContext))
        {
            auto resultRight = mlirGen(binaryExpressionAST->right, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(resultRight)
            auto resultRightValue = V(resultRight);
            
            return mlirGenCallThisMethod(location, resultRightValue, SYMBOL_HAS_INSTANCE, undefined, {binaryExpressionAST->left}, genContext);
        }        

        auto resultLeft = mlirGen(binaryExpressionAST->left, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(resultLeft)
        auto resultLeftValue = V(resultLeft);

        auto resultLeftfType = resultLeftValue.getType();
        if (auto refType = dyn_cast<mlir_ts::RefType>(resultLeftfType))
        {
            resultLeftfType = refType.getElementType();
        }

        resultLeftfType = mth.wideStorageType(resultLeftfType);

        // TODO: should it be mlirGen?
        auto resultRight = mlirGen(binaryExpressionAST->right, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(resultRight)
        auto resultRightValue = V(resultRight);

        auto rightType = resultRightValue.getType();
        if (mth.isNoneType(rightType))
        {
            emitError(location, "type of instanceOf can't be resolved.");
            return mlir::failure();
        }

        rightType = mth.wideStorageType(rightType);

#ifdef ENABLE_RTTI
        if (auto classType = dyn_cast<mlir_ts::ClassType>(rightType))
        {
            if (isa<mlir_ts::ClassType>(resultLeftfType))
            {
                NodeFactory nf(NodeFactoryFlags::None);
                NodeArray<Expression> argumentsArray;
                argumentsArray.push_back(nf.createPropertyAccessExpression(binaryExpressionAST->right, nf.createIdentifier(S(RTTI_NAME))));
                return mlirGenCallThisMethod(location, resultLeftValue, INSTANCEOF_NAME, undefined, argumentsArray, genContext);
            }

            if (isa<mlir_ts::AnyType>(resultLeftfType))
            {
                auto typeOfAnyValue = builder.create<mlir_ts::TypeOfAnyOp>(location, getStringType(), resultLeftValue);
                auto classStrConst =
                    builder.create<mlir_ts::ConstantOp>(location, getStringType(), builder.getStringAttr("class"));
                auto cmpResult = builder.create<mlir_ts::StringCompareOp>(
                    location, getBooleanType(), typeOfAnyValue, classStrConst,
                    builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken));

                MLIRCodeLogicHelper mclh(builder, location);
                auto returnValue = mclh.conditionalExpression(
                    getBooleanType(), cmpResult,
                    [&](mlir::OpBuilder &builder, mlir::Location location) {
                        // TODO: test cast value
                        auto thisPtrValue = cast(location, getOpaqueType(), resultLeftValue, genContext);
                        return mlirGenInstanceOfOpaque(location, thisPtrValue, resultRightValue, genContext);
                    },
                    [&](mlir::OpBuilder &builder, mlir::Location location) { // default false value
                                                                             // compare typeOfValue
                        return builder.create<mlir_ts::ConstantOp>(location, getBooleanType(),
                                                                   builder.getBoolAttr(false));
                    });

                return returnValue;
            }

            if (isa<mlir_ts::OpaqueType>(resultLeftfType))
            {
                return mlirGenInstanceOfOpaque(location, resultLeftValue, resultRightValue, genContext);
            }
        }
#endif

        LLVM_DEBUG(llvm::dbgs() << "!! instanceOf precalc value: " << (resultLeftfType == rightType) << " '" << resultLeftfType
                                << "' is '" << rightType << "'\n";);

        // default logic
        return V(
            builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(resultLeftfType == rightType)));
    }

    ValueOrLogicalResult evaluateBinaryOp(mlir::Location location, SyntaxKind opCode, mlir_ts::ConstantOp leftConstOp,
                                 mlir_ts::ConstantOp rightConstOp, const GenContext &genContext)
    {
        // todo string concat
        auto leftStrAttr = dyn_cast_or_null<mlir::StringAttr>(leftConstOp.getValueAttr());
        auto rightStrAttr = dyn_cast_or_null<mlir::StringAttr>(rightConstOp.getValueAttr());        
        if (leftStrAttr && rightStrAttr)
        {
            auto leftStr = leftStrAttr.getValue();
            auto rightStr = rightStrAttr.getValue();

            std::string result;
            switch (opCode)
            {
                case SyntaxKind::PlusToken:
                    result = leftStr;
                    result += rightStr;
                    break;
                default:
                    emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                    return mlir::failure();
            }

            return V(builder.create<mlir_ts::ConstantOp>(location, getStringType(), builder.getStringAttr(result)));
        }

        auto leftIntAttr = dyn_cast_or_null<mlir::IntegerAttr>(leftConstOp.getValueAttr());
        auto rightIntAttr = dyn_cast_or_null<mlir::IntegerAttr>(rightConstOp.getValueAttr());
        auto resultType = leftConstOp.getType();
        if (leftIntAttr && rightIntAttr)
        {
            auto leftInt = leftIntAttr.getValue();
            auto rightInt = rightIntAttr.getValue();            
            auto result = leftInt;
            switch (opCode)
            {
            case SyntaxKind::PlusToken:
                result = leftInt + rightInt;
                break;
            case SyntaxKind::MinusToken:
                result = leftInt - rightInt;
                break;
            case SyntaxKind::AsteriskToken:
                result = leftInt * rightInt;
                break;
            case SyntaxKind::LessThanLessThanToken:
                result = leftInt << rightInt;
                break;
            case SyntaxKind::GreaterThanGreaterThanToken:
                result = leftInt.ashr(rightInt);
                break;
            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                result = leftInt.lshr(rightInt);
                break;
            case SyntaxKind::AmpersandToken:
                result = leftInt & rightInt;
                break;
            case SyntaxKind::BarToken:
                result = leftInt | rightInt;
                break;
            case SyntaxKind::CaretToken:
                result = leftInt ^ rightInt;
                break;
            default:
                emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                return mlir::failure();
            }

            return V(builder.create<mlir_ts::ConstantOp>(location, resultType, builder.getIntegerAttr(leftIntAttr.getType(), result)));
        }

        auto leftFloatAttr = dyn_cast_or_null<mlir::FloatAttr>(leftConstOp.getValueAttr());
        auto rightFloatAttr = dyn_cast_or_null<mlir::FloatAttr>(rightConstOp.getValueAttr());
        if (leftFloatAttr && rightFloatAttr)
        {
            auto leftFloat = leftFloatAttr.getValue();
            auto rightFloat = rightFloatAttr.getValue();
            auto result = leftFloat;

            auto useSigned = true;
            APSInt leftAPInt(64, /*isUnsigned=*/!useSigned);
            APSInt rightAPInt(64, /*isUnsigned=*/!useSigned);
            APSInt resultAPInt(64, /*isUnsigned=*/!useSigned);

            bool ignored;
            auto castStatus = APFloat::opInvalidOp == leftFloat.convertToInteger(leftAPInt, APFloat::rmTowardZero, &ignored);
            if (castStatus)
            {
                emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                return mlir::failure();
            }

            castStatus = APFloat::opInvalidOp == rightFloat.convertToInteger(rightAPInt, APFloat::rmTowardZero, &ignored);
            if (castStatus)
            {
                emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                return mlir::failure();
            }

            switch (opCode)
            {
            case SyntaxKind::PlusToken:
                result = leftFloat + rightFloat;
                break;
            case SyntaxKind::MinusToken:
                result = leftFloat - rightFloat;
                break;
            case SyntaxKind::AsteriskToken:
                result = leftFloat * rightFloat;
                break;
            case SyntaxKind::LessThanLessThanToken:
                resultAPInt = leftAPInt.shl(rightAPInt);
                break;
            case SyntaxKind::GreaterThanGreaterThanToken:
                resultAPInt = leftAPInt.ashr(rightAPInt);
                break;
            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                resultAPInt = leftAPInt.lshr(rightAPInt);
                break;
            case SyntaxKind::AmpersandToken:
                resultAPInt = leftAPInt & rightAPInt;
                break;
            case SyntaxKind::BarToken:
                resultAPInt = leftAPInt | rightAPInt;
                break;
            case SyntaxKind::CaretToken:
                resultAPInt = leftAPInt ^ rightAPInt;
                break;
            default:
                emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                return mlir::failure();
            }

            switch (opCode)
            {
            case SyntaxKind::PlusToken:
            case SyntaxKind::MinusToken:
            case SyntaxKind::AsteriskToken:
                break;
            default:
                castStatus = APFloat::opInvalidOp == result.convertFromAPInt(resultAPInt, /*IsSigned=*/useSigned,
                        APFloat::rmNearestTiesToEven);
                if (castStatus)
                {
                    emitError(location) << "can't do binary operation on constants: " << leftConstOp.getValueAttr() << " and " << rightConstOp.getValueAttr() << "";
                    return mlir::failure();
                }
                break;
            }            

            auto resultAttr = builder.getFloatAttr(leftFloatAttr.getType(), result);
            return V(builder.create<mlir_ts::ConstantOp>(location, resultType, resultAttr));
        }    

        return mlir::failure();    
    }

    ValueOrLogicalResult mlirGenSaveLogicOneItem(mlir::Location location, mlir::Value leftExpressionValue,
                                                 mlir::Value rightExpressionValue, const GenContext &genContext)
    {
        if (!leftExpressionValue)
        {
            return mlir::failure();
        }

        auto leftExpressionValueBeforeCast = leftExpressionValue;

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            if (isa<mlir_ts::CharType>(rightExpressionValue.getType()))
            {
                CAST(rightExpressionValue, location, getStringType(), rightExpressionValue, genContext);
            }
        }

        auto savingValue = rightExpressionValue;
        if (!savingValue)
        {
            return mlir::failure();
        }

        auto syncSavingValue = [&](mlir::Type destType) {
            if (destType != savingValue.getType())
            {
                savingValue = cast(location, destType, savingValue, genContext);
            }
        };

        // TODO: finish it for field access, review CodeLogicHelper.saveResult
        if (auto loadOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::LoadOp>())
        {
            mlir::Type destType =
                mlir::TypeSwitch<mlir::Type, mlir::Type>(loadOp.getReference().getType())
                    .Case<mlir_ts::RefType>([&](auto refType) { return refType.getElementType(); })
                    .Case<mlir_ts::BoundRefType>([&](auto boundRefType) { return boundRefType.getElementType(); });

            assert(destType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! Dest type: " << destType << "\n";);

            syncSavingValue(destType);
            if (!savingValue)
            {
                return mlir::failure();
            }

            // TODO: when saving const array into variable we need to allocate space and copy array as we need to have
            // writable array
            builder.create<mlir_ts::StoreOp>(location, savingValue, loadOp.getReference());
        }
        else if (auto extractPropertyOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ExtractPropertyOp>())
        {
            syncSavingValue(extractPropertyOp.getType());
            if (!savingValue)
            {
                return mlir::failure();
            }

            // access to conditional tuple
            // let's see if we can get reference to it
            MLIRCodeLogic mcl(builder);
            auto propRef = mcl.GetReferenceFromValue(location, leftExpressionValueBeforeCast);
            if (!propRef)
            {
                return mlir::failure();
            }

            builder.create<mlir_ts::StoreOp>(location, savingValue, propRef);
            return mlir::success();                

        }
        else if (auto accessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::AccessorOp>())
        {
            syncSavingValue(accessorOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::AccessorOp>(
                location, mlir::Type(),
                accessorOp.getGetAccessorAttr(), 
                accessorOp.getSetAccessorAttr(), 
                savingValue);            
        }
        else if (auto thisAccessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisAccessorOp>())
        {
            syncSavingValue(thisAccessorOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::ThisAccessorOp>(
                location, mlir::Type(), thisAccessorOp.getThisVal(),
                thisAccessorOp.getGetAccessorAttr(), 
                thisAccessorOp.getSetAccessorAttr(), 
                savingValue);
        }
        else if (auto thisAccessorIndirectOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisIndirectAccessorOp>())
        {
            syncSavingValue(thisAccessorIndirectOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // TODO: it should return accessor as result as it will return data
            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::ThisIndirectAccessorOp>(
                location, mlir::Type(), 
                thisAccessorIndirectOp.getThisVal(), 
                thisAccessorIndirectOp.getGetAccessor(), 
                thisAccessorIndirectOp.getSetAccessor(), 
                savingValue);    
        } 
        else if (auto thisIndexAccessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisIndexAccessorOp>())
        {
            syncSavingValue(thisIndexAccessorOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::ThisIndexAccessorOp>(
                location, mlir::Type(), thisIndexAccessorOp.getThisVal(), thisIndexAccessorOp.getIndex(),
                thisIndexAccessorOp.getGetAccessorAttr(), 
                thisIndexAccessorOp.getSetAccessorAttr(), 
                savingValue);
        }         
        else if (auto thisIndirectIndexAccessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisIndirectIndexAccessorOp>())
        {
            syncSavingValue(thisIndirectIndexAccessorOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::ThisIndirectIndexAccessorOp>(
                location, mlir::Type(), thisIndirectIndexAccessorOp.getThisVal(), thisIndirectIndexAccessorOp.getIndex(),
                thisIndirectIndexAccessorOp.getGetAccessor(), 
                thisIndirectIndexAccessorOp.getSetAccessor(), 
                savingValue);
        }                
        else if (auto boundAccessorIndirectOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::BoundIndirectAccessorOp>())
        {
            syncSavingValue(boundAccessorIndirectOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // TODO: it should return accessor as result as it will return data
            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::BoundIndirectAccessorOp>(
                location, mlir::Type(), 
                boundAccessorIndirectOp.getGetAccessor(), 
                boundAccessorIndirectOp.getSetAccessor(), 
                savingValue);    
        }         
        else if (auto boundIndirectIndexAccessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::BoundIndirectIndexAccessorOp>())
        {
            syncSavingValue(boundIndirectIndexAccessorOp.getType(0));
            if (!savingValue)
            {
                return mlir::failure();
            }

            // we create new instance of accessor with saving value, previous will be deleted as not used
            auto callRes = builder.create<mlir_ts::BoundIndirectIndexAccessorOp>(
                location, mlir::Type(), boundIndirectIndexAccessorOp.getIndex(),
                boundIndirectIndexAccessorOp.getGetAccessor(), 
                boundIndirectIndexAccessorOp.getSetAccessor(), 
                savingValue);
        }           
        /*
        else if (auto createBoundFunction = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            // TODO: i should not allow to change interface
            return mlirGenSaveLogicOneItem(location, createBoundFunction.getFunc(), rightExpressionValue, genContext);
        }
        */
        else if (auto lengthOf = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::LengthOfOp>())
        {
            MLIRCodeLogic mcl(builder);
            auto arrayValueLoaded = mcl.GetReferenceFromValue(location, lengthOf.getOp());
            if (!arrayValueLoaded)
            {
                emitError(location) << "Can't get reference of the array, ensure const array is not used";
                return mlir::failure();
            }

            // special case to resize array
            syncSavingValue(lengthOf.getResult().getType());
            builder.create<mlir_ts::SetLengthOfOp>(location, arrayValueLoaded, savingValue);
        }
        else if (auto stringLength = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::StringLengthOp>())
        {
            MLIRCodeLogic mcl(builder);
            auto stringValueLoaded = mcl.GetReferenceFromValue(location, stringLength.getOp());
            if (!stringValueLoaded)
            {
                emitError(location) << "Can't get reference of the string, ensure const string is not used";
                return mlir::failure();
            }

            // special case to resize array
            syncSavingValue(stringLength.getResult().getType());
            builder.create<mlir_ts::SetStringLengthOp>(location, stringValueLoaded, savingValue);
        }
        else
        {
            LLVM_DEBUG(dbgs() << "\n!! left expr.: " << leftExpressionValueBeforeCast << " ...\n";);
            emitError(location, "saving to constant object");
            return mlir::failure();
        }

        return savingValue;
    }

    ValueOrLogicalResult mlirGenSaveLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (leftExpression == SyntaxKind::ArrayLiteralExpression)
        {
            return mlirGenSaveLogicArray(location, leftExpression.as<ArrayLiteralExpression>(), rightExpression,
                                         genContext);
        }

        if (leftExpression == SyntaxKind::ObjectLiteralExpression)
        {
            return mlirGenSaveLogicObject(location, leftExpression.as<ObjectLiteralExpression>(), rightExpression,
                                          genContext);
        }

        auto result = mlirGen(leftExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto leftExpressionValue = V(result);

        auto rightExprGenContext = GenContext(genContext);
        rightExprGenContext.clearReceiverTypes();

        if (mth.isAnyFunctionType(leftExpressionValue.getType()))
        {
            rightExprGenContext.receiverFuncType = leftExpressionValue.getType();
        }

        rightExprGenContext.receiverType = leftExpressionValue.getType();

        auto result2 = mlirGen(rightExpression, rightExprGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result2)
        auto rightExpressionValue = V(result2);

        return mlirGenSaveLogicOneItem(location, leftExpressionValue, rightExpressionValue, genContext);
    }

    ValueOrLogicalResult mlirGenSaveLogicArray(mlir::Location location, ArrayLiteralExpression arrayLiteralExpression,
                                               Expression rightExpression, const GenContext &genContext)
    {
        auto result = mlirGen(rightExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto rightExpressionValue = V(result);

        LLVM_DEBUG(dbgs() << "\n!! right expr.: " << rightExpressionValue << "\n";);

        auto isTuple = false;
        mlir::Type elementType;
        mlir_ts::TupleType tupleType;
        mlir::TypeSwitch<mlir::Type>(rightExpressionValue.getType())
            .Case<mlir_ts::ArrayType>([&](auto arrayType) { elementType = arrayType.getElementType(); })
            .Case<mlir_ts::ConstArrayType>([&](auto constArrayType) { elementType = constArrayType.getElementType(); })
            .Case<mlir_ts::TupleType>([&](auto tupleType_) { isTuple = true; tupleType = tupleType_; })
            .Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { isTuple = true; tupleType = mth.convertConstTupleTypeToTupleType(constTupleType); })
            .Default([](auto type) { llvm_unreachable("not implemented"); });

        if (!isTuple)
        {
            for (auto [index, leftItem] : enumerate(arrayLiteralExpression->elements))
            {
                auto result = mlirGen(leftItem, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto leftExpressionValue = V(result);

                // special case for [a = 1, b = 2] = [2, 3];
                if (leftItem == SyntaxKind::BinaryExpression)
                {
                    auto binExpr = leftItem.as<BinaryExpression>();
                    auto result = mlirGen(binExpr->left, genContext);
                    EXIT_IF_FAILED_OR_NO_VALUE(result)
                    leftExpressionValue = V(result);
                }

                // TODO: unify array access like Property access
                auto indexValue =
                    builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(index));

                auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType),
                                                                    rightExpressionValue, indexValue);
                auto rightValue = builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);

                if (mlir::failed(mlirGenSaveLogicOneItem(location, leftExpressionValue, rightValue, genContext)))
                {
                    return mlir::failure();
                }
            }
        }
        else
        {
            for (auto [index, leftItem] : enumerate(arrayLiteralExpression->elements))
            {
                auto result = mlirGen(leftItem, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto leftExpressionValue = V(result);

                // special case for [a = 1, b = "abc"] = [2, "def"];
                if (leftItem == SyntaxKind::BinaryExpression)
                {
                    auto binExpr = leftItem.as<BinaryExpression>();
                    auto result = mlirGen(binExpr->left, genContext);
                    EXIT_IF_FAILED_OR_NO_VALUE(result)
                    leftExpressionValue = V(result);
                }

                MLIRPropertyAccessCodeLogic cl(builder, location, rightExpressionValue, builder.getI32IntegerAttr(index));
                auto rightValue = cl.Tuple(tupleType, true);
                if (!rightValue)
                {
                    return mlir::failure();
                }

                if (mlir::failed(mlirGenSaveLogicOneItem(location, leftExpressionValue, rightValue, genContext)))
                {
                    return mlir::failure();
                }
            }

        }

        // no passing value
        return mlir::success();
    }

    ValueOrLogicalResult mlirGenSaveLogicObject(mlir::Location location,
                                                ObjectLiteralExpression objectLiteralExpression,
                                                Expression rightExpression, const GenContext &genContext)
    {
        auto result = mlirGen(rightExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto rightExpressionValue = V(result);

        for (auto item : objectLiteralExpression->properties)
        {
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();

                auto propertyName = MLIRHelper::getName(propertyAssignment->name);

                auto result = mlirGen(propertyAssignment->initializer, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto ident = V(result);

                auto subInit =
                    mlirGenPropertyAccessExpression(location, rightExpressionValue, propertyName, false, genContext);

                if (mlir::failed(mlirGenSaveLogicOneItem(location, ident, subInit, genContext)))
                {
                    return mlir::failure();
                }
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();

                auto propertyName = MLIRHelper::getName(shorthandPropertyAssignment->name);
                auto varName = propertyName;

                auto ident = resolveIdentifier(location, varName, genContext);

                auto subInit =
                    mlirGenPropertyAccessExpression(location, rightExpressionValue, propertyName, false, genContext);

                if (mlir::failed(mlirGenSaveLogicOneItem(location, ident, subInit, genContext)))
                {
                    return mlir::failure();
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        // no passing value
        return mlir::success();
    }

    mlir::LogicalResult unwrapForBinaryOp(mlir::Location location, SyntaxKind opCode, mlir::Value &leftExpressionValue,
                                          mlir::Value &rightExpressionValue, const GenContext &genContext)
    {
        if (opCode == SyntaxKind::CommaToken)
        {
            return mlir::success();
        }

        // type preprocess
        // TODO: temporary hack
        if (auto leftType = dyn_cast<mlir_ts::LiteralType>(leftExpressionValue.getType()))
        {
            CAST(leftExpressionValue, location, leftType.getElementType(), leftExpressionValue, genContext);
        }

        if (auto rightType = dyn_cast<mlir_ts::LiteralType>(rightExpressionValue.getType()))
        {
            CAST(rightExpressionValue, location, rightType.getElementType(), rightExpressionValue, genContext);
        }
        // end of hack

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            // TODO: temporary hack
            if (isa<mlir_ts::CharType>(leftExpressionValue.getType()))
            {
                CAST(leftExpressionValue, location, getStringType(), leftExpressionValue, genContext);
            }

            if (isa<mlir_ts::CharType>(rightExpressionValue.getType()))
            {
                CAST(rightExpressionValue, location, getStringType(), rightExpressionValue, genContext);
            }

            // end todo

            if (!MLIRLogicHelper::isLogicOp(opCode))
            {
                // TODO: review it
                // cast from optional<T> type
                if (auto leftOptType = dyn_cast<mlir_ts::OptionalType>(leftExpressionValue.getType()))
                {
                    leftExpressionValue =
                        builder.create<mlir_ts::ValueOrDefaultOp>(location, leftOptType.getElementType(), leftExpressionValue);
                }

                if (auto rightOptType = dyn_cast<mlir_ts::OptionalType>(rightExpressionValue.getType()))
                {
                    rightExpressionValue =
                        builder.create<mlir_ts::ValueOrDefaultOp>(location, rightOptType.getElementType(), rightExpressionValue);
                }
            }
        }
        else if (!MLIRLogicHelper::isLogicOp(opCode))
        {
            // TODO: review it
            // special case both are optionals
            if (auto leftOptType = dyn_cast<mlir_ts::OptionalType>(leftExpressionValue.getType()))
            {
                if (auto rightOptType = dyn_cast<mlir_ts::OptionalType>(rightExpressionValue.getType()))
                {
                    leftExpressionValue =
                        builder.create<mlir_ts::ValueOrDefaultOp>(location, leftOptType.getElementType(), leftExpressionValue);
                    rightExpressionValue =
                        builder.create<mlir_ts::ValueOrDefaultOp>(location, rightOptType.getElementType(), rightExpressionValue);
                }
            }
        }

        return mlir::success();
    }

    bool syncTypes(mlir::Location location, mlir::Type type, mlir::Value & leftExpressionValue, mlir::Value & rightExpressionValue, const GenContext &genContext)
    {
        auto hasType = leftExpressionValue.getType() == type ||
                            rightExpressionValue.getType() == type;
        if (hasType)
        {
            if (leftExpressionValue.getType() != type)
            {
                CAST(leftExpressionValue, location, type, leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != type)
            {
                CAST(rightExpressionValue, location, type, rightExpressionValue, genContext);
            }

            return true;
        }

        return false;
    }

    mlir::Type SInt(int width) 
    {
        return mlir::IntegerType::get(builder.getContext(), width, mlir::IntegerType::Signed);
    }

    // TODO: review it, seems like big hack
    mlir::LogicalResult adjustTypesForBinaryOp(mlir::Location location, SyntaxKind opCode, mlir::Value &leftExpressionValue,
                                               mlir::Value &rightExpressionValue, const GenContext &genContext)
    {
        if (opCode == SyntaxKind::CommaToken)
        {
            return mlir::success();
        }

        // cast step
        switch (opCode)
        {
        case SyntaxKind::CommaToken:
            // no cast needed
            break;
        case SyntaxKind::LessThanLessThanToken:
        case SyntaxKind::GreaterThanGreaterThanToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
        case SyntaxKind::AmpersandToken:
        case SyntaxKind::BarToken:
        case SyntaxKind::CaretToken:
            // cast to int
            if (leftExpressionValue.getType() != builder.getI32Type())
            {
                CAST(leftExpressionValue, location, builder.getI32Type(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != builder.getI32Type())
            {
                CAST(rightExpressionValue, location, builder.getI32Type(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::SlashToken:
        case SyntaxKind::PercentToken:
        case SyntaxKind::AsteriskAsteriskToken:

            if (leftExpressionValue.getType() != getNumberType())
            {
                CAST(leftExpressionValue, location, getNumberType(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != getNumberType())
            {
                CAST(rightExpressionValue, location, getNumberType(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::AsteriskToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:

            if (isa<mlir_ts::UndefinedType>(leftExpressionValue.getType()) || isa<mlir_ts::UndefinedType>(rightExpressionValue.getType()))
            {
                break;
            }

            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                static SmallVector<mlir::Type> types = {builder.getF128Type(), getNumberType(), builder.getF64Type(), builder.getI64Type(), SInt(64), builder.getF32Type(), SInt(32), 
                    builder.getI32Type(), builder.getF16Type(), SInt(16), builder.getI16Type(), SInt(8), builder.getI8Type()};
                for (auto type : types)
                {
                    if (syncTypes(location, type, leftExpressionValue, rightExpressionValue, genContext))
                    {
                        break;
                    }
                }
            }

            break;
        default:
            auto resultType = leftExpressionValue.getType();
            if (isa<mlir_ts::StringType>(rightExpressionValue.getType()))
            {
                resultType = getStringType();
                if (resultType != leftExpressionValue.getType())
                {
                    CAST(leftExpressionValue, location, resultType, leftExpressionValue, genContext);
                }
            }

            if (resultType != rightExpressionValue.getType())
            {
                CAST(rightExpressionValue, location, resultType, rightExpressionValue, genContext);
            }

            break;
        }

        return mlir::success();
    }

    mlir::Value binaryOpLogic(mlir::Location location, SyntaxKind opCode, mlir::Value leftExpressionValue,
                              mlir::Value rightExpressionValue, const GenContext &genContext)
    {
        auto result = rightExpressionValue;
        switch (opCode)
        {
        case SyntaxKind::EqualsToken:
            // nothing to do;
            assert(false);
            break;
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:
            result = builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                                                              builder.getI32IntegerAttr((int)opCode),
                                                              leftExpressionValue, rightExpressionValue);
            break;
        case SyntaxKind::CommaToken:
            return rightExpressionValue;
        default:
            result = builder.create<mlir_ts::ArithmeticBinaryOp>(location, leftExpressionValue.getType(),
                                                                 builder.getI32IntegerAttr((int)opCode),
                                                                 leftExpressionValue, rightExpressionValue);
            break;
        }

        return result;
    }

    ValueOrLogicalResult mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto opCode = (SyntaxKind)binaryExpressionAST->operatorToken;

        auto saveResult = MLIRLogicHelper::isNeededToSaveData(opCode);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
        {
            return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken,
                                     saveResult);
        }

        if (opCode == SyntaxKind::QuestionQuestionToken)
        {
            return mlirGenQuestionQuestionLogic(binaryExpressionAST, saveResult, genContext);
        }

        if (opCode == SyntaxKind::InKeyword)
        {
            return mlirGenInLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::InstanceOfKeyword)
        {
            return mlirGenInstanceOfLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::EqualsToken)
        {
            return mlirGenSaveLogic(binaryExpressionAST, genContext);
        }

        auto result = mlirGen(leftExpression, genContext);
        if (opCode == SyntaxKind::CommaToken)
        {
            //in case of "commad" op the result of left op can be "nothing"
            EXIT_IF_FAILED(result)
        }
        else
        {
            EXIT_IF_FAILED_OR_NO_VALUE(result)    
        }

        auto leftExpressionValue = V(result);
        auto result2 = mlirGen(rightExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result2)
        auto rightExpressionValue = V(result2);

        // check if const expr.
        if (genContext.allowConstEval)
        {
            LLVM_DEBUG(llvm::dbgs() << "Evaluate const: '" << leftExpressionValue << "' and '" << rightExpressionValue << "'\n";);

            auto leftConstOp = dyn_cast<mlir_ts::ConstantOp>(leftExpressionValue.getDefiningOp());
            auto rightConstOp = dyn_cast<mlir_ts::ConstantOp>(rightExpressionValue.getDefiningOp());
            if (leftConstOp && rightConstOp)
            {
                // try to evaluate
                return evaluateBinaryOp(location, opCode, leftConstOp, rightConstOp, genContext);
            }
        }

        auto leftExpressionValueBeforeCast = leftExpressionValue;
        auto rightExpressionValueBeforeCast = rightExpressionValue;

        unwrapForBinaryOp(location, opCode, leftExpressionValue, rightExpressionValue, genContext);

        adjustTypesForBinaryOp(location, opCode, leftExpressionValue, rightExpressionValue, genContext);

        auto resultReturn = binaryOpLogic(location, opCode, leftExpressionValue, rightExpressionValue, genContext);

        if (saveResult)
        {
            return mlirGenSaveLogicOneItem(location, leftExpressionValueBeforeCast, resultReturn, genContext);
        }

        return resultReturn;
    }

    ValueOrLogicalResult mlirGen(SpreadElement spreadElement, const GenContext &genContext)
    {
        return mlirGen(spreadElement->expression, genContext);
    }

    ValueOrLogicalResult mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
    {
        return mlirGen(parenthesizedExpression->expression, genContext);
    }

    ValueOrLogicalResult mlirGen(QualifiedName qualifiedName, const GenContext &genContext)
    {
        auto location = loc(qualifiedName);

        auto expression = qualifiedName->left;
        auto result = mlirGenModuleReference(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        auto name = MLIRHelper::getName(qualifiedName->right);

        return mlirGenPropertyAccessExpression(location, expressionValue, name, genContext);
    }

    ValueOrLogicalResult mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
    {
        auto location = loc(propertyAccessExpression);

        auto expression = propertyAccessExpression->expression.as<Expression>();
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        auto namePtr = MLIRHelper::getName(propertyAccessExpression->name, stringAllocator);

        return mlirGenPropertyAccessExpression(location, expressionValue, namePtr,
                                               !!propertyAccessExpression->questionDotToken, genContext);
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, name);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, bool isConditional,
                                                         const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, name);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, id);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, id);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         mlir::Value argument/*for index access*/,
                                                         const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, id, argument);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }    

    ValueOrLogicalResult mlirGenPropertyAccessExpressionLogic(mlir::Location location, mlir::Value objectValue,
                                                              bool isConditional, MLIRPropertyAccessCodeLogic &cl,
                                                              const GenContext &genContext)
    {
        if (isConditional && mth.isNullableOrOptionalType(objectValue.getType()))
        {
            // TODO: replace with one op "Optional <has_value>, <value>"
            CAST_A(condValue, location, getBooleanType(), objectValue, genContext);

            auto propType = evaluateProperty(location, objectValue, cl.getName().str(), genContext);
            if (!propType)
            {
                emitError(location, "Can't resolve property '") << cl.getName() << "' of type " << to_print(objectValue.getType());
                return mlir::failure();
            }

            auto ifOp = builder.create<mlir_ts::IfOp>(location, getOptionalType(propType), condValue, true);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

            // value if true
            auto result = mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
            auto value = V(result);

            // special case: conditional extension function <xxx>?.<ext>();
            if (auto createExtentionFunction = value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
            {
                // we need to convert into CreateBoundFunction, so it should be reference type for this, do I need to case value type into reference type?
                value = createBoundMethodFromExtensionMethod(location, createExtentionFunction);
                ifOp.getResults().front().setType(getOptionalType(value.getType()));
            }

            auto optValue = isa<mlir_ts::OptionalType>(value.getType())
                    ? value : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});

            // else
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

            auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, getOptionalType(value.getType()));
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.getResults().front();
        }
        else
        {
            return mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
        }
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpressionBaseLogic(mlir::Location location, mlir::Value objectValue,
                                                                  MLIRPropertyAccessCodeLogic &cl,
                                                                  const GenContext &genContext)
    {
        auto name = cl.getName();
        auto argument = cl.getArgument();
        auto actualType = objectValue.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n\tResolving property '" << name << "' of type " << objectValue.getType(););

        // load reference if needed, except TupleTuple, ConstTupleType
        if (auto refType = dyn_cast<mlir_ts::RefType>(actualType))
        {
            auto elementType = refType.getElementType();
            if (!isa<mlir_ts::TupleType>(elementType) && !isa<mlir_ts::ConstTupleType>(elementType))
            {
                objectValue = builder.create<mlir_ts::LoadOp>(location, elementType, objectValue);
                actualType = objectValue.getType();
            }
        }

        // class member access
        auto classAccessWithObject = [&](mlir_ts::ClassType classType, mlir::Value objectValue) {
            if (auto value = cl.Class(classType))
            {
                return value;
            }

            return ClassMembers(location, objectValue, classType.getName().getValue(), name, 
                false, argument, genContext);
        };

        auto classAccess = [&](mlir_ts::ClassType classType) {
            return classAccessWithObject(classType, objectValue);
        };

        mlir::Value value = 
            mlir::TypeSwitch<mlir::Type, mlir::Value>(actualType)
                .Case<mlir_ts::EnumType>([&](auto enumType) { return cl.Enum(enumType); })
                .Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { return cl.Tuple(constTupleType); })
                .Case<mlir_ts::TupleType>([&](auto tupleType) { return cl.Tuple(tupleType); })
                .Case<mlir_ts::BooleanType>([&](auto intType) { 
                    if (auto value = cl.Bool(intType))
                    {
                        return value;
                    }
                    
                    return mlir::Value();                    
                })
                .Case<mlir::IntegerType>([&](auto intType) { return cl.Int(intType); })
                .Case<mlir::FloatType>([&](auto floatType) { return cl.Float(floatType); })
                .Case<mlir_ts::NumberType>([&](auto numberType) {                    
                    if (auto value = cl.Number(numberType))
                    {
                        return value;
                    }

                    return mlir::Value();                        
                })
                .Case<mlir_ts::StringType>([&](auto stringType) { 
                    if (auto value = cl.String(stringType))
                    {
                        return value;
                    }

                    return mlir::Value();
                })
                .Case<mlir_ts::ConstArrayType>([&](auto arrayType) { 
                    if (auto genericClassTypeInfo = getGenericClassInfoByFullName("Array"))
                    {
                        auto classType = genericClassTypeInfo->classType;
                        SmallVector<mlir::Type> typeArg{arrayType.getElementType()};
                        auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                typeArg, genContext, true);
                        auto accessFailed = false;
                        if (mlir::succeeded(result))
                        {
                            auto arrayNonConst = cast(location, mlir_ts::ArrayType::get(arrayType.getElementType()), objectValue, genContext);
                            if (arrayNonConst.failed())
                            {
                                return mlir::Value();
                            }

                            if (auto value = classAccessWithObject(mlir::cast<mlir_ts::ClassType>(specType), arrayNonConst))
                            {
                                return value;
                            }

                            accessFailed = true;
                        }

                        if (mlir::failed(result) && !accessFailed)
                        {
                            const_cast<GenContext &>(genContext).stop();
                            return mlir::Value();
                        }

                        genContext.postponedMessages->clear();
                    }

                    // find Array type
                    // TODO: should I mix use of Array and Array<T>?
                    // if (auto classInfo = getClassInfoByFullName("Array"))
                    // {
                    //     return classAccess(classInfo->classType);
                    // }

                    if (auto value = cl.Array(arrayType))
                    {
                        return value;
                    }
                    
                    return mlir::Value();   
                })
                .Case<mlir_ts::ArrayType>([&](auto arrayType) { 
                    if (auto genericClassTypeInfo = getGenericClassInfoByFullName("Array"))
                    {
                        auto classType = genericClassTypeInfo->classType;
                        SmallVector<mlir::Type> typeArg{arrayType.getElementType()};
                        auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                typeArg, genContext, true);
                        auto accessFailed = false;
                        if (mlir::succeeded(result))
                        {
                            if (auto value = classAccess(mlir::cast<mlir_ts::ClassType>(specType)))
                            {
                                return value;
                            }

                            accessFailed = true;
                        }

                        if (mlir::failed(result) && !accessFailed)
                        {
                            const_cast<GenContext &>(genContext).stop();
                            return mlir::Value();
                        }

                        genContext.postponedMessages->clear();
                    }

                    // find Array type
                    // TODO: should I mix use of Array and Array<T>?
                    // if (auto classInfo = getClassInfoByFullName("Array"))
                    // {
                    //     return classAccess(classInfo->classType);
                    // }

                    if (auto value = cl.Array(arrayType))
                    {
                        return value;
                    }
                    
                    return mlir::Value();                      
                })
                .Case<mlir_ts::RefType>([&](auto refType) { return cl.Ref(refType); })
                .Case<mlir_ts::ObjectType>([&](auto objectType) { 
                    if (auto value = cl.Object(objectType))
                    {
                        return value;
                    }

                    return mlir::Value();                    
                })
                .Case<mlir_ts::ObjectStorageType>([&](auto objectStorageType) { 
                    if (auto value = cl.RefLogic(objectStorageType))
                    {
                        return value;
                    }

                    return mlir::Value();                    
                })
                .Case<mlir_ts::SymbolType>([&](auto symbolType) { return cl.Symbol(symbolType); })
                .Case<mlir_ts::NamespaceType>([&](auto namespaceType) {
                    auto namespaceInfo = getNamespaceByFullName(namespaceType.getName().getValue());
                    assert(namespaceInfo);

                    MLIRNamespaceGuard ng(currentNamespace);
                    currentNamespace = namespaceInfo;

                    return mlirGen(location, name, genContext);
                })
                .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                    if (auto value = cl.TupleNoError(classStorageType))
                    {
                        return value;
                    }

                    return ClassMembers(location, objectValue, 
                        classStorageType.getName().getValue(), name, true, argument, genContext);
                })
                .Case<mlir_ts::ClassType>(classAccess)
                .Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    return InterfaceMembers(
                        location, objectValue, interfaceType.getName().getValue(), cl.getAttribute(), 
                        argument, genContext);
                })
                .Case<mlir_ts::OptionalType>([&](auto optionalType) {
                    // this is needed for conditional access to properties
                    auto elementType = optionalType.getElementType();
                    auto loadedValue = builder.create<mlir_ts::ValueOp>(location, elementType, objectValue);
                    return mlirGenPropertyAccessExpression(location, loadedValue, name, false, genContext);                
                })
                .Case<mlir_ts::UnionType>([&](auto unionType) {
                    // TODO: when access of property in union is finished use it instead of using first type
                    // all union types must have the same property
                    // 1) cast to first type
                    auto frontType = mth.getFirstNonNullUnionType(unionType);
                    //auto casted = cast(location, frontType, objectValue, genContext);
                    auto casted = builder.create<mlir_ts::GetValueFromUnionOp>(location, frontType, objectValue);

                    return mlirGenPropertyAccessExpression(location, casted, name, false, genContext);
                })
                .Case<mlir_ts::LiteralType>([&](auto literalType) {
                    auto elementType = literalType.getElementType();
                    auto castedValue = builder.create<mlir_ts::CastOp>(location, elementType, objectValue);
                    return mlirGenPropertyAccessExpression(location, castedValue, name, false, genContext);
                })
                .Default([&](auto type) {
                    LLVM_DEBUG(llvm::dbgs() << "\n\tCan't resolve property '" << name << "' of type " << objectValue.getType(););
                    return mlir::Value();
                });

        // extention logic: <obj>.<functionName>(this)
        if (!value)
        {
            if (auto funcRef = extensionFunction(location, objectValue, name, genContext))
            {
                return funcRef;
            }
        }

        if (!value)
        {
            emitError(location, "Can't resolve property '") << name << "' of type " << to_print(objectValue.getType());
            return mlir::failure();
        }

        return value;
    }

    mlir::Value extensionFunctionLogic(mlir::Location location, mlir::Value funcRef, mlir::Value thisValue, StringRef name,
                                  const GenContext &genContext)
    {
        if (!mth.isAnyFunctionType(funcRef.getType()))
        {
            return mlir::Value();
        }

        LLVM_DEBUG(llvm::dbgs() << "!! found extension by name for type: " << thisValue.getType()
                                << " function: " << name << ", value: " << funcRef << "\n";);

        auto thisTypeFromFunc = mth.getFirstParamFromFuncRef(funcRef.getType());

        LLVM_DEBUG(llvm::dbgs() << "!! this type of function is : " << thisTypeFromFunc << "\n";);

        if (auto symbolOp = funcRef.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            // if (!isa<mlir_ts::GenericType>(symbolOp.getType()))
            if (!symbolOp->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME))
            {
                auto funcType = mlir::cast<mlir_ts::FunctionType>(funcRef.getType());
                if (thisTypeFromFunc == thisValue.getType())
                {
                    // return funcRef;
                    auto thisRef = thisValue;
                    auto extensFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                        location, getExtensionFunctionType(funcType), thisRef, funcRef);
                    return extensFuncVal;
                }
            }
            else
            {
                // TODO: finish it
                // it is generic function
                StringMap<mlir::Type> inferredTypes;
                inferType(location, thisTypeFromFunc, thisValue.getType(), inferredTypes, genContext);
                if (inferredTypes.size() > 0)
                {
                    // we found needed function
                    // return funcRef;
                    auto thisRef = thisValue;

                    LLVM_DEBUG(llvm::dbgs() << "\n!! recreate ExtensionFunctionOp (generic interface): '" << name << "'\n this ref: '" << thisRef << "'\n func ref: '" << funcRef
                    << "'\n";);

                    auto funcType = mlir::cast<mlir_ts::FunctionType>(funcRef.getType());
                    auto extensFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                        location, getExtensionFunctionType(funcType), thisRef, funcRef);
                    return extensFuncVal;                        
                }
            }
        }

        return mlir::Value();
    }

    mlir::Value extensionFunction(mlir::Location location, mlir::Value thisValue, StringRef name,
                                  const GenContext &genContext)
    {
        if (auto funcRef = resolveIdentifier(location, name, genContext))
        {
            auto result = extensionFunctionLogic(location, funcRef, thisValue, name, genContext);
            if (result)
            {
                return result;
            }
        }

        // look into all namespaces from current one
        {
            MLIRNamespaceGuard ng(currentNamespace);

            // search in outer namespaces
            while (currentNamespace->isFunctionNamespace)
            {
                currentNamespace = currentNamespace->parentNamespace;
            }

            auto &currentNamespacesMap = currentNamespace->namespacesMap;
            for (auto &selectedNamespace : currentNamespacesMap)
            {
                currentNamespace = selectedNamespace.getValue();
                if (auto funcRef = resolveIdentifierInNamespace(location, name, genContext))
                {
                    auto result = extensionFunctionLogic(location, funcRef, thisValue, name, genContext);
                    if (result)
                    {
                        return result;
                    }
                }
            }
        }        

        return mlir::Value();
    }

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, mlir::StringRef classFullName,
                             mlir::StringRef name, bool baseClass, mlir::Value argument, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classFullName);
        if (!classInfo)
        {
            auto genericClassInfo = getGenericClassInfoByFullName(classFullName);
            if (genericClassInfo)
            {
                // we can't discover anything in generic class
                return mlir::Value();
            }

            emitError(location, "Class can't be found ") << classFullName;
            return mlir::Value();
        }

        // static field access
        auto value = ClassMembers(location, thisValue, classInfo, name, baseClass, argument, genContext);
        if (!value)
        {
            emitError(location, "Class member '") << name << "' can't be found";
        }

        return value;
    }

    mlir::Value getThisRefOfClass(mlir::Location location, mlir_ts::ClassType classType, mlir::Value thisValue, bool isSuperClass, const GenContext &genContext)
    {
        auto effectiveThisValue = thisValue;
        if (isSuperClass)
        {
            // LLVM_DEBUG(dbgs() << "\n!! base call: func '" << funcOp.getName() << "' in context func. '"
            //                     << const_cast<GenContext &>(genContext).funcOp.getName()
            //                     << "', this type: " << thisValue.getType() << " value:" << thisValue << "";);

            // get reference in case of classStorage
            auto isStorageType = isa<mlir_ts::ClassStorageType>(thisValue.getType());
            if (isStorageType)
            {
                MLIRCodeLogic mcl(builder);
                thisValue = mcl.GetReferenceFromValue(location, thisValue);
                assert(thisValue);
            }

            CAST(effectiveThisValue, location, classType, thisValue, genContext);
        }        

        return effectiveThisValue;
    }

    mlir::Value ClassStaticFieldAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int staticFieldIndex, const GenContext &genContext) {

        auto fieldInfo = classInfo->staticFields[staticFieldIndex];
#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
        if (thisValue.getDefiningOp<mlir_ts::ClassRefOp>() || classInfo->isStatic)
        {
#endif
            auto value = resolveFullNameIdentifier(location, fieldInfo.globalVariableName, false, genContext);
            // load referenced value
            if (classInfo->isDynamicImport)
            {
                if (auto valueRefType = dyn_cast<mlir_ts::RefType>(value.getType()))
                {
                    value = builder.create<mlir_ts::LoadOp>(location, valueRefType.getElementType(), value);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }

            return value;
#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
        }

        // static accessing via class reference
        // TODO:
        auto effectiveThisValue = thisValue;

        auto result = mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);
        auto vtableAccess = V(result);

        assert(genContext.allowPartialResolve || fieldInfo.virtualIndex >= 0);

        auto virtualSymbOp = builder.create<mlir_ts::VirtualSymbolRefOp>(
            location, mlir_ts::RefType::get(fieldInfo.type), vtableAccess,
            builder.getI32IntegerAttr(fieldInfo.virtualIndex),
            mlir::FlatSymbolRefAttr::get(builder.getContext(), fieldInfo.globalVariableName));

        auto value = builder.create<mlir_ts::LoadOp>(location, fieldInfo.type, virtualSymbOp);
        return value;
#endif
    }

    mlir::Value ClassMethodAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int methodIndex, bool isSuperClass, const GenContext &genContext) {

        LLVM_DEBUG(llvm::dbgs() << "\n!! method index access: " << methodIndex << "\n";);

        auto methodInfo = classInfo->methods[methodIndex];
        auto funcOp = methodInfo.funcOp;
        auto effectiveFuncType = funcOp.getFunctionType();

        if (methodInfo.isStatic)
        {
#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
            auto isThisValueClassRef = thisValue.getDefiningOp<mlir_ts::ClassRefOp>();
            if (isThisValueClassRef || classInfo->isStatic)
            {
#endif
                if (classInfo->isDynamicImport)
                {
                    // need to resolve global variable
                    auto globalFuncVar = resolveFullNameIdentifier(location, funcOp.getName(), false, genContext);

                    if (!isThisValueClassRef)
                    {
                        CAST_A(opaqueThisValue, location, getOpaqueType(), thisValue, genContext);
                        auto boundMethodValue = builder.create<mlir_ts::CreateBoundFunctionOp>(
                            location, getBoundFunctionType(effectiveFuncType), opaqueThisValue, globalFuncVar);  
                        return boundMethodValue;                          
                    }

                    return globalFuncVar;
                }
                else
                {
                    if (!isThisValueClassRef)
                    {
                        auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                            location, getBoundFunctionType(effectiveFuncType), thisValue,
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));                            
                        
                        return thisSymbOp;
                    }

                    auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                        location, effectiveFuncType,
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                    return symbOp;
                }
#ifdef ADD_STATIC_MEMBERS_TO_VTABLE
            }

            // static accessing via class reference
            // TODO:
            auto effectiveThisValue = thisValue;

            auto vtableAccess =
                mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);

            if (!vtableAccess)
            {
                emitError(location,"") << "class '" << classInfo->fullName << "' missing 'virtual table'";
            }

            EXIT_IF_FAILED_OR_NO_VALUE(vtableAccess)                    

            assert(genContext.allowPartialResolve || methodInfo.virtualIndex >= 0);

            auto virtualSymbOp = builder.create<mlir_ts::VirtualSymbolRefOp>(
                location, effectiveFuncType, vtableAccess, builder.getI32IntegerAttr(methodInfo.virtualIndex),
                mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
            return virtualSymbOp;
#endif
        }
        else
        {
            auto effectiveThisValue = getThisRefOfClass(location, classInfo->classType, thisValue, isSuperClass, genContext);

            // TODO: check if you can split calls such as "this.method" and "super.method" ...
            auto isStorageType = isa<mlir_ts::ClassStorageType>(thisValue.getType());
            if (methodInfo.isAbstract || /*!baseClass &&*/ methodInfo.isVirtual && !isStorageType)
            {
                LLVM_DEBUG(dbgs() << "\n!! Virtual call: func '" << funcOp.getName() << "'\n";);

                LLVM_DEBUG(dbgs() << "\n!! Virtual call - this val: [ " << effectiveThisValue << " ] func type: [ "
                                    << effectiveFuncType << " ] isStorage access: " << isStorageType << "\n";);

                // auto inTheSameFunc = funcOp.getName() == const_cast<GenContext &>(genContext).funcOp.getName();

                auto vtableAccess =
                    mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);

                if (!vtableAccess)
                {
                    emitError(location,"") << "class '" << classInfo->fullName << "' missing 'virtual table'";
                }

                EXIT_IF_FAILED_OR_NO_VALUE(vtableAccess)

                assert(genContext.allowPartialResolve || methodInfo.virtualIndex >= 0);

                auto thisVirtualSymbOp = builder.create<mlir_ts::ThisVirtualSymbolRefOp>(
                    location, getBoundFunctionType(effectiveFuncType), effectiveThisValue, vtableAccess,
                    builder.getI32IntegerAttr(methodInfo.virtualIndex),
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return thisVirtualSymbOp;
            }

            if (classInfo->isDynamicImport)
            {
                // need to resolve global variable
                auto globalFuncVar = resolveFullNameIdentifier(location, funcOp.getName(), false, genContext);
                CAST_A(opaqueThisValue, location, getOpaqueType(), effectiveThisValue, genContext);
                auto boundMethodValue = builder.create<mlir_ts::CreateBoundFunctionOp>(
                    location, getBoundFunctionType(effectiveFuncType), opaqueThisValue, globalFuncVar);
                return boundMethodValue;
            }
            else
            {
                // default call;
                auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                    location, getBoundFunctionType(effectiveFuncType), effectiveThisValue,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return thisSymbOp;
            }
        }
    }    

    mlir::Value ClassGenericMethodAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int genericMethodIndex, 
            bool isSuperClass, const GenContext &genContext) {
        auto genericMethodInfo = classInfo->staticGenericMethods[genericMethodIndex];

        auto paramsArray = genericMethodInfo.funcProto->getParams();
        auto explicitThis = paramsArray.size() > 0 && paramsArray.front()->getName() == THIS_NAME;
        if (genericMethodInfo.isStatic && !explicitThis)
        {
            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, genericMethodInfo.funcType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericMethodInfo.funcProto->getName()));
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return funcSymbolOp;
        }
        else
        {
            auto effectiveThisValue = getThisRefOfClass(location, classInfo->classType, thisValue, isSuperClass, genContext);
            auto effectiveFuncType = genericMethodInfo.funcProto->getFuncType();

            auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                location, getBoundFunctionType(effectiveFuncType), effectiveThisValue,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericMethodInfo.funcProto->getName()));
            thisSymbOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return thisSymbOp;                
        }
    }

    mlir::Value ClassAccessorAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int accessorIndex, const GenContext &genContext) {

        auto accessorInfo = classInfo->accessors[accessorIndex];
        auto getFuncOp = accessorInfo.get;
        auto setFuncOp = accessorInfo.set;
        mlir::Type accessorResultType;
        if (getFuncOp)
        {
            auto funcType = dyn_cast<mlir_ts::FunctionType>(getFuncOp.getFunctionType());
            if (funcType.getNumResults() > 0)
            {
                accessorResultType = funcType.getResult(0);
            }
        }

        if (!accessorResultType && setFuncOp)
        {
            accessorResultType =
                dyn_cast<mlir_ts::FunctionType>(setFuncOp.getFunctionType()).getInput(accessorInfo.isStatic ? 0 : 1);
        }

        if (!accessorResultType)
        {
            emitError(location) << "can't resolve type of property";
            return mlir::Value();
        }

        if (accessorInfo.isStatic)
        {
            auto accessorOp = builder.create<mlir_ts::AccessorOp>(
                location, accessorResultType,
                getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName())
                            : mlir::FlatSymbolRefAttr{},
                setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName())
                            : mlir::FlatSymbolRefAttr{},
                mlir::Value());
            return accessorOp.getResult(0);
        }
        else
        {
            auto thisAccessorOp = builder.create<mlir_ts::ThisAccessorOp>(
                location, accessorResultType, thisValue,
                getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName())
                            : mlir::FlatSymbolRefAttr{},
                setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName())
                            : mlir::FlatSymbolRefAttr{},
                mlir::Value());
            return thisAccessorOp.getResult(0);
        }

    }

    // TODO: why isSuperClass is not used here?
    mlir::Value ClassIndexAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, mlir::Value argument, const GenContext &genContext) {

        if (classInfo->indexes.size() == 0)
        {
            emitError(location) << "indexer is not declared";
            return mlir::Value();            
        }

        auto indexInfo = classInfo->indexes.front();
        auto getFuncOp = indexInfo.get;
        auto setFuncOp = indexInfo.set;

        if (!indexInfo.indexSignature || indexInfo.indexSignature.getNumResults() == 0)
        {
            emitError(location) << "can't resolve type of indexer";
            return mlir::Value();
        }

        auto indexResultType = indexInfo.indexSignature.getResult(0);
        auto argumentType = indexInfo.indexSignature.getInput(0);

        // sync index
        CAST_A(result, location, argumentType, argument, genContext);

        auto thisIndexAccessorOp = builder.create<mlir_ts::ThisIndexAccessorOp>(
            location, indexResultType, thisValue, V(result),
            getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName())
                        : mlir::FlatSymbolRefAttr{},
            setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName())
                        : mlir::FlatSymbolRefAttr{},
            mlir::Value());
        return thisIndexAccessorOp.getResult(0);
    }

    mlir::Value ClassBaseClassAccess(ClassInfo::TypePtr classInfo, ClassInfo::TypePtr baseClass, int index,
            mlir::Location location, mlir::Value thisValue, StringRef name, mlir::Value argument, const GenContext &genContext) {

        // first base is "super."
        if (index == 0 && name == SUPER_NAME)
        {
            auto result = mlirGenPropertyAccessExpression(location, thisValue, baseClass->fullName, genContext);
            auto value = V(result);
            return value;
        }

        auto value = ClassMembers(location, thisValue, baseClass, name, true, argument, genContext);
        if (value)
        {
            return value;
        }

        SmallVector<ClassInfo::TypePtr> fieldPath;
        if (classHasField(baseClass, name, fieldPath))
        {
            // load value from path
            auto currentObject = thisValue;
            for (auto &chain : fieldPath)
            {
                auto fieldValue =
                    mlirGenPropertyAccessExpression(location, currentObject, chain->fullName, genContext);
                if (!fieldValue)
                {
                    emitError(location) << "Can't resolve field/property/base '" << chain->fullName
                                        << "' of class '" << classInfo->fullName << "'\n";
                    return fieldValue;
                }

                assert(fieldValue);
                currentObject = fieldValue;
            }

            // last value
            auto result = mlirGenPropertyAccessExpression(location, currentObject, name, genContext);
            auto value = V(result);
            if (value)
            {
                return value;
            }
        }

        return mlir::Value();
    }    

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, ClassInfo::TypePtr classInfo,
                             mlir::StringRef name, bool isSuperClass, mlir::Value argument, const GenContext &genContext)
    {
        assert(classInfo);

        LLVM_DEBUG(llvm::dbgs() << "\n\t looking for member: " << name << " in class '" << classInfo->fullName << "\n";);

        // indexer access
        if (name == INDEX_ACCESS_FIELD_NAME)
        {
            return ClassIndexAccess(classInfo, location, thisValue, argument, genContext);
        }

        auto staticFieldIndex = classInfo->getStaticFieldIndex(
            MLIRHelper::TupleFieldName(name, builder.getContext()));
        if (staticFieldIndex >= 0)
        {
            return ClassStaticFieldAccess(classInfo, location, thisValue, staticFieldIndex, genContext);
        }

        // check method access
        auto methodIndex = classInfo->getMethodIndex(name);
        if (methodIndex >= 0)
        {
            return ClassMethodAccess(classInfo, location, thisValue, methodIndex, isSuperClass, genContext);
        }

        // static generic methods
        auto genericMethodIndex = classInfo->getGenericMethodIndex(name);
        if (genericMethodIndex >= 0)
        {        
            return ClassGenericMethodAccess(classInfo, location, thisValue, genericMethodIndex, isSuperClass, genContext);
        }        

        // check accessor
        auto accessorIndex = classInfo->getAccessorIndex(name);
        if (accessorIndex >= 0)
        {
            return ClassAccessorAccess(classInfo, location, thisValue, accessorIndex, genContext);
        }

        for (auto [index, baseClass] : enumerate(classInfo->baseClasses))
        {
            auto value = ClassBaseClassAccess(classInfo, baseClass, index, location, 
                thisValue, name, argument, genContext);
            if (value)
            {
                return value;
            }
        }

        if (isSuperClass || genContext.allowPartialResolve)
        {
            return mlir::Value();
        }

        emitError(location) << "can't resolve property/field/base '" << name << "' of class '" << classInfo->fullName
                            << "'\n";

        return mlir::Value();
    }

    bool classHasField(ClassInfo::TypePtr classInfo, mlir::StringRef name, SmallVector<ClassInfo::TypePtr> &fieldPath)
    {
        auto fieldId = MLIRHelper::TupleFieldName(name, builder.getContext());
        auto classStorageType = mlir::cast<mlir_ts::ClassStorageType>(classInfo->classType.getStorageType());
        auto fieldIndex = classStorageType.getIndex(fieldId);
        auto missingField = fieldIndex < 0 || fieldIndex >= classStorageType.size();
        if (!missingField)
        {
            fieldPath.insert(fieldPath.begin(), classInfo);
            return true;
        }

        for (auto baseClass : classInfo->baseClasses)
        {
            if (classHasField(baseClass, name, fieldPath))
            {
                fieldPath.insert(fieldPath.begin(), classInfo);
                return true;
            }
        }

        return false;
    }

    mlir::Value InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, mlir::StringRef interfaceFullName,
                                 mlir::Attribute id, mlir::Value argument, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceFullName);
        if (!interfaceInfo)
        {
            auto genericInterfaceInfo = getGenericInterfaceInfoByFullName(interfaceFullName);
            if (genericInterfaceInfo)
            {
                // we can't detect value of generic interface (we can only if it is specialization)
                emitError(location, "Interface can't be found ") << interfaceFullName;
                return mlir::Value();
            }

            return mlir::Value();
        }

        assert(interfaceInfo);

        // static field access
        auto value = InterfaceMembers(location, interfaceValue, interfaceInfo, id, argument, genContext);
        if (!value)
        {
            emitError(location, "Interface member '") << id << "' can't be found";
        }

        return value;
    }

    mlir::Value InterfaceFieldAccess(mlir::Location location, mlir::Value interfaceValue, InterfaceFieldInfo *fieldInfo) 
    {
        auto fieldRefType = mlir_ts::RefType::get(fieldInfo->type);
        if (fieldInfo->virtualIndex == -1)
        {
            // no data for conditional interface;
            if (!fieldInfo->isConditional)
            {
                emitError(location, "field '") << fieldInfo->id << "' is not conditional and missing";
                return mlir::Value();
            }

            auto actualType = isa<mlir_ts::OptionalType>(fieldRefType.getElementType())
                                    ? fieldRefType.getElementType()
                                    : mlir_ts::OptionalType::get(fieldRefType.getElementType());
            return builder.create<mlir_ts::OptionalUndefOp>(location, actualType);
        }

        assert(fieldInfo->virtualIndex >= 0);
        auto vtableIndex = fieldInfo->virtualIndex;

        auto interfaceSymbolRefValue = builder.create<mlir_ts::InterfaceSymbolRefOp>(
            location, fieldRefType, interfaceValue, builder.getI32IntegerAttr(vtableIndex),
            fieldInfo->id, builder.getBoolAttr(fieldInfo->isConditional));

        mlir::Value value;
        if (!fieldInfo->isConditional)
        {
            value = builder.create<mlir_ts::LoadOp>(location, fieldRefType.getElementType(),
                                                    interfaceSymbolRefValue.getResult());
        }
        else
        {
            auto actualType = isa<mlir_ts::OptionalType>(fieldRefType.getElementType())
                                    ? fieldRefType.getElementType()
                                    : mlir_ts::OptionalType::get(fieldRefType.getElementType());
            value = builder.create<mlir_ts::LoadOp>(location, actualType, interfaceSymbolRefValue.getResult());
        }

        // if it is FuncType, we need to create BoundMethod again
        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(fieldInfo->type))
        {
            auto thisVal =
                builder.create<mlir_ts::ExtractInterfaceThisOp>(location, getOpaqueType(), interfaceValue);
            value = builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType),
                                                                    thisVal, value);
        }

        return value;
    }

    mlir::Value InterfaceMethodAccess(mlir::Location location, mlir::Value interfaceValue, InterfaceMethodInfo *methodInfo) 
    {
        assert(methodInfo->virtualIndex >= 0);
        auto vtableIndex = methodInfo->virtualIndex;

        auto effectiveFuncType = getBoundFunctionType(methodInfo->funcType);

        auto interfaceSymbolRefValue = builder.create<mlir_ts::InterfaceSymbolRefOp>(
            location, effectiveFuncType, interfaceValue, builder.getI32IntegerAttr(vtableIndex),
            builder.getStringAttr(methodInfo->name), builder.getBoolAttr(methodInfo->isConditional));

        return interfaceSymbolRefValue;
    }    

    mlir::Value InterfaceAccessorAccess(mlir::Location location, InterfaceInfo::TypePtr interfaceInfo, 
            mlir::Value interfaceValue, InterfaceAccessorInfo *accessorInfo, const GenContext &genContext) {

        assert(accessorInfo);

        mlir::Value getMethodInfoValue;
        mlir::Value setMethodInfoValue;
        if (!accessorInfo->getMethod.empty())
        {
            if (auto getMethodInfo = interfaceInfo->findMethod(accessorInfo->getMethod))
            {
                getMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, getMethodInfo);
            }
            else
            {
                emitError(location) << "Can't find method " << accessorInfo->getMethod << " in interface '" << to_print(interfaceInfo->interfaceType) << "'";
                return mlir::Value();
            }
        }
        else
        {
            getMethodInfoValue = builder.create<mlir_ts::UndefOp>(location, getBoundFunctionType({}, {}, false));
        }

        if (!accessorInfo->setMethod.empty())
        {
            if (auto setMethodInfo = interfaceInfo->findMethod(accessorInfo->setMethod))
            {
                setMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, setMethodInfo);
            }
            else
            {
                emitError(location) << "Can't find method " << accessorInfo->setMethod << " in interface '" << to_print(interfaceInfo->interfaceType) << "'";
                return mlir::Value();
            }
        }
        else
        {
            setMethodInfoValue = builder.create<mlir_ts::UndefOp>(location, getBoundFunctionType({}, {}, false));
        }

        auto thisIndirectAccessorOp = builder.create<mlir_ts::BoundIndirectAccessorOp>(
            location, accessorInfo->type, getMethodInfoValue, setMethodInfoValue,
            mlir::Value());

        LLVM_DEBUG(llvm::dbgs() << "\n!! .... : " << thisIndirectAccessorOp << "\n";);

        assert(thisIndirectAccessorOp.getGetAccessor());
        
        return thisIndirectAccessorOp.getResult(0);
    }

    mlir::Value InterfaceIndexAccess(InterfaceInfo::TypePtr interfaceInfo, 
            mlir::Location location, mlir::Value interfaceValue, mlir::Value argument, const GenContext &genContext) {

        auto indexInfo = interfaceInfo->findIndexer();

        if (!indexInfo)
        {
            emitError(location) << "indexer is not declared";
            return mlir::Value();            
        }

        if (!indexInfo->indexSignature || indexInfo->indexSignature.getNumResults() == 0)
        {
            emitError(location) << "can't resolve type of indexer";
            return mlir::Value();
        }

        auto [argumentType, indexResultType] = mth.getIndexSignatureArgumentAndResultTypes(indexInfo->indexSignature);

        // sync index
        CAST_A(result, location, argumentType, argument, genContext);

        mlir::Value getMethodInfoValue;
        mlir::Value setMethodInfoValue;
        if (!indexInfo->getMethod.empty())
        {
            if (auto getMethodInfo = interfaceInfo->findMethod(indexInfo->getMethod))
            {
                getMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, getMethodInfo);
            }
            else
            {
                emitError(location) << "Can't find method " << INDEX_ACCESS_GET_FIELD_NAME << " in interface '" << to_print(interfaceInfo->interfaceType) << "'";
                return mlir::Value();
            }
        }
        else
        {
            getMethodInfoValue = builder.create<mlir_ts::UndefOp>(location, getBoundFunctionType({}, {}, false));
        }

        if (!indexInfo->setMethod.empty())
        {
            if (auto setMethodInfo = interfaceInfo->findMethod(indexInfo->setMethod))
            {
                setMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, setMethodInfo);
            }
            else
            {
                emitError(location) << "Can't find method " << INDEX_ACCESS_SET_FIELD_NAME << " in interface '" << to_print(interfaceInfo->interfaceType) << "'";
                return mlir::Value();
            }
        }
        else
        {
            setMethodInfoValue = builder.create<mlir_ts::UndefOp>(location, getBoundFunctionType({}, {}, false));
        }

        auto thisIndirectIndexAccessorOp = builder.create<mlir_ts::BoundIndirectIndexAccessorOp>(
            location, indexResultType, V(result), getMethodInfoValue, setMethodInfoValue,
            mlir::Value());
        return thisIndirectIndexAccessorOp.getResult(0);
    }

    mlir::Value InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, InterfaceInfo::TypePtr interfaceInfo, 
        mlir::Attribute id, mlir::Value argument, const GenContext &genContext)
    {
        assert(interfaceInfo);

        // indexer access
        auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(id);
        if (nameAttr && nameAttr.getValue() == INDEX_ACCESS_FIELD_NAME)
        {
            return InterfaceIndexAccess(interfaceInfo, location, interfaceValue, argument, genContext);
        }

        // check field access        
        if (auto fieldInfo = interfaceInfo->findField(id))
        {
            return InterfaceFieldAccess(location, interfaceValue, fieldInfo);
        }

        // check method access
        if (nameAttr)
        {
            if (auto methodInfo = interfaceInfo->findMethod(nameAttr.getValue()))
            {
                return InterfaceMethodAccess(location, interfaceValue, methodInfo);
            }

            if (auto accessorInfo = interfaceInfo->findAccessor(nameAttr.getValue()))
            {
                return InterfaceAccessorAccess(location, interfaceInfo, interfaceValue, accessorInfo, genContext);
            }

        }

        return mlir::Value();
    }

    template <typename T>
    ValueOrLogicalResult mlirGenElementAccessTuple(mlir::Location location, mlir::Value expression,
                                              mlir::Value argumentExpression, T tupleType)
    {
        // get index
        if (auto indexConstOp = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
        {
            // this is property access
            MLIRPropertyAccessCodeLogic cl(builder, location, expression, indexConstOp.getValue());
            return cl.Tuple(tupleType, true);
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! index value: " << argumentExpression
                                    << ", check if tuple must be an array\n";);
            llvm_unreachable("not implemented (index)");
        }
    }

    ValueOrLogicalResult mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext)
    {
        auto location = loc(elementAccessExpression);

        auto conditinalAccess = !!elementAccessExpression->questionDotToken;

        auto result = mlirGen(elementAccessExpression->expression.as<Expression>(), genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expression = V(result);

        // default access <array>[index]
        if (!conditinalAccess)
        {
            auto result2 = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result2)
            auto argumentExpression = V(result2);

            return mlirGenElementAccess(location, expression, argumentExpression, conditinalAccess, genContext);
        }

        // <array>?.[index] access
        CAST_A(condValue, location, getBooleanType(), expression, genContext);
        return conditionalValue(location, condValue, 
            [&](auto genContext) { 
                auto result2 = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result2)
                auto argumentExpression = V(result2);

                // conditinalAccess should be false here
                auto result3 = mlirGenElementAccess(location, expression, argumentExpression, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result3)
                auto value = V(result3);

                auto optValue = 
                    isa<mlir_ts::OptionalType>(value.getType())
                        ? value
                        : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                return ValueOrLogicalResult(optValue); 
            }, 
            [&](mlir::Type trueValueType, auto genContext) { 
                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, trueValueType);
                return ValueOrLogicalResult(optUndefValue); 
            }, 
            genContext);
    }

    ValueOrLogicalResult mlirGenElementAccess(mlir::Location location, mlir::Value expression, mlir::Value argumentExpression, bool isConditionalAccess, const GenContext &genContext)
    {
        auto arrayType = expression.getType();
        if (isa<mlir_ts::LiteralType>(arrayType))
        {
            arrayType = mth.stripLiteralType(arrayType);
            CAST(expression, location, arrayType, expression, genContext);
        }

        if (auto optType = dyn_cast<mlir_ts::OptionalType>(arrayType))
        {
            arrayType = optType.getElementType();
            // loading value from opt value
            expression = builder.create<mlir_ts::ValueOp>(location, arrayType, expression);
        }

        mlir::Type elementType;
        if (auto arrayTyped = dyn_cast<mlir_ts::ArrayType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr)) 
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            elementType = arrayTyped.getElementType();
        }
        else if (auto vectorType = dyn_cast<mlir_ts::ConstArrayType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr)) 
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            elementType = vectorType.getElementType();
        }
        else if (isa<mlir_ts::StringType>(arrayType))
        {
            elementType = getCharType();
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(arrayType))
        {
            return mlirGenElementAccessTuple(location, expression, argumentExpression, tupleType);
        }
        else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(arrayType))
        {
            return mlirGenElementAccessTuple(location, expression, argumentExpression, constTupleType);
        }
        else if (auto classType = dyn_cast<mlir_ts::ClassType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (attr.isa<mlir::StringAttr>())
                {
                    // TODO: implement '[string]' access here
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            // else access of index
            auto indexAccessor = builder.getStringAttr(INDEX_ACCESS_FIELD_NAME);
            return mlirGenPropertyAccessExpression(location, expression, indexAccessor, isConditionalAccess, argumentExpression, genContext);
        }
        else if (auto classStorageType = dyn_cast<mlir_ts::ClassStorageType>(arrayType))
        {
            // seems we are calling "super"
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
            }

            llvm_unreachable("not implemented (ElementAccessExpression)");
        }        
        else if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (attr.isa<mlir::StringAttr>())
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            // else access of index
            auto indexAccessor = builder.getStringAttr(INDEX_ACCESS_FIELD_NAME);
            return mlirGenPropertyAccessExpression(location, expression, indexAccessor, isConditionalAccess, argumentExpression, genContext);
        }        
        else if (auto enumType = dyn_cast<mlir_ts::EnumType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
            }

            llvm_unreachable("not implemented (ElementAccessExpression)");
        }          
        else if (auto refType = dyn_cast<mlir_ts::RefType>(arrayType)) 
        {
            CAST_A(index, location, mth.getStructIndexType(), argumentExpression, genContext);

            auto elemRef = builder.create<mlir_ts::PointerOffsetRefOp>(
                location, refType, expression, index);            

            return V(elemRef);
        }
        else if (auto anyType = dyn_cast<mlir_ts::AnyType>(arrayType))
        {
            emitError(location, "not supported");
            return mlir::failure();
        }          
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! ElementAccessExpression: " << arrayType
                                    << "\n";);

            emitError(location) << "access expression is not applicable to " << to_print(arrayType);
            return mlir::failure();
        }

        auto indexType = argumentExpression.getType();
        CAST(argumentExpression, location, mth.getStructIndexType(), argumentExpression, genContext);
  
        auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression,
                                                             argumentExpression);
        return V(builder.create<mlir_ts::LoadOp>(location, elementType, elemRef));
    }

    ValueOrLogicalResult mlirGen(CallExpression callExpression, const GenContext &genContext)
    {
        auto location = loc(callExpression);

        auto callExpr = callExpression->expression.as<Expression>();

        auto result = mlirGen(callExpr, genContext);
        // in case of detecting value for recursive calls we need to ignore failed calls
        if (result.failed_or_no_value() && genContext.allowPartialResolve)
        {
            // we need to return success to continue code traversing
            return V(builder.create<mlir_ts::UndefOp>(location, builder.getNoneType()));
        }

        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto funcResult = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!! evaluate function: " << funcResult << "\n";);

        auto funcType = funcResult.getType();
        if (!mth.isAnyFunctionType(funcType) 
            && !mth.isBuiltinFunctionType(funcResult)
            // TODO: do I need to use ConstructFunction instead?
            // to support constructor calls
            && !isa<mlir_ts::ClassType>(funcType)
            // to support super.constructor calls
            && !isa<mlir_ts::ClassStorageType>(funcType))
        {           
            // TODO: rewrite code for calling "5.ToString()"
            // TODO: recursive functions are usually return "failure" as can't be found
            //return mlir::failure();
            return funcResult;
        }

        // so if method is generic and you need to infer types you can cast to generic types
        auto noReceiverTypesForGenericCall = 
            mth.isGenericType(funcResult.getType()) 
            && callExpression->typeArguments.size() == 0;

        SmallVector<mlir::Value, 4> operands;
        auto offsetArgs = isa<mlir_ts::BoundFunctionType>(funcType) || isa<mlir_ts::ExtensionFunctionType>(funcType) ? 1 : 0;
        if (mlir::failed(mlirGenOperands(callExpression->arguments, operands, funcResult.getType(), genContext, offsetArgs, noReceiverTypesForGenericCall)))
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! function: [" << funcResult << "] ops: "; for (auto o
                                                                                       : operands) llvm::dbgs()
                                                                                  << "\n param type: " << o.getType();
                   llvm::dbgs() << "\n";);

        return mlirGenCallExpression(location, funcResult, callExpression->typeArguments, operands, genContext);
    }

    mlir::LogicalResult mlirGenArrayForEach(mlir::Location location, ArrayRef<mlir::Value> operands,
                                            const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto arraySrc = operands[0];
        auto funcSrc = operands[1];

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto funcVarDecl = std::make_shared<VariableDeclarationDOM>(".func", funcSrc.getType(), location);
        DECLARE(funcVarDecl, funcSrc);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _func_ident = nf.createIdentifier(S(".func"));

        auto _v_ident = nf.createIdentifier(S(".v"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto forOfStat = nf.createForOfStatement(
            undefined, declList, _src_array_ident,
            nf.createExpressionStatement(nf.createCallExpression(_func_ident, undefined, argumentsArray)));

        mlirGen(forOfStat, genContext);

        return mlir::success();
    }

    ValueOrLogicalResult mlirGenArrayEvery(mlir::Location location, ArrayRef<mlir::Value> operands,
                                           const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto varName = ".ev";
        auto initVal = builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(true));
        registerVariable(
            location, varName, false, VariableType::Let,
            [&](mlir::Location, const GenContext &) -> TypeValueInitType {
                return {getBooleanType(), initVal, TypeProvided::No};
            },
            genContext);

        auto arraySrc = operands[0];
        auto funcSrc = operands[1];

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto funcVarDecl = std::make_shared<VariableDeclarationDOM>(".func", funcSrc.getType(), location);
        DECLARE(funcVarDecl, funcSrc);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _func_ident = nf.createIdentifier(S(".func"));

        auto _v_ident = nf.createIdentifier(S(".v"));
        auto _result_ident = nf.createIdentifier(stows(varName));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto forOfStat = nf.createForOfStatement(
            undefined, declList, _src_array_ident,
            nf.createIfStatement(
                nf.createPrefixUnaryExpression(
                    nf.createToken(SyntaxKind::ExclamationToken),
                    nf.createBinaryExpression(_result_ident, nf.createToken(SyntaxKind::AmpersandAmpersandEqualsToken),
                                              nf.createCallExpression(_func_ident, undefined, argumentsArray))),
                nf.createBreakStatement(), undefined));

        mlirGen(forOfStat, genContext);

        return resolveIdentifier(location, varName, genContext);
    }

    ValueOrLogicalResult mlirGenArraySome(mlir::Location location, ArrayRef<mlir::Value> operands,
                                          const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto varName = ".sm";
        auto initVal = builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(false));
        registerVariable(
            location, varName, false, VariableType::Let,
            [&](mlir::Location, const GenContext &) -> TypeValueInitType {
                return {getBooleanType(), initVal, TypeProvided::No};
            },
            genContext);

        auto arraySrc = operands[0];
        auto funcSrc = operands[1];

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto funcVarDecl = std::make_shared<VariableDeclarationDOM>(".func", funcSrc.getType(), location);
        DECLARE(funcVarDecl, funcSrc);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _func_ident = nf.createIdentifier(S(".func"));

        auto _v_ident = nf.createIdentifier(S(".v"));
        auto _result_ident = nf.createIdentifier(stows(varName));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto forOfStat = nf.createForOfStatement(
            undefined, declList, _src_array_ident,
            nf.createIfStatement(
                nf.createBinaryExpression(_result_ident, nf.createToken(SyntaxKind::BarBarEqualsToken),
                                          nf.createCallExpression(_func_ident, undefined, argumentsArray)),
                nf.createBreakStatement(), undefined));

        mlirGen(forOfStat, genContext);

        return resolveIdentifier(location, varName, genContext);
    }

    ValueOrLogicalResult mlirGenArrayMap(mlir::Location location, ArrayRef<mlir::Value> operands,
                                         const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto arraySrc = operands[0];
        auto funcSrc = operands[1];

        auto [pos, _end] = LocationHelper::getSpan(location);

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto funcVarDecl = std::make_shared<VariableDeclarationDOM>(".func", funcSrc.getType(), location);
        DECLARE(funcVarDecl, funcSrc);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _func_ident = nf.createIdentifier(S(".func"));

        auto _v_ident = nf.createIdentifier(S(".v"));
        
        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto _yield_expr = nf.createYieldExpression(undefined,
            nf.createCallExpression(_func_ident, undefined, argumentsArray));
        _yield_expr->pos.pos = pos;
        _yield_expr->_end = _end;

        auto forOfStat =
            nf.createForOfStatement(undefined, declList, _src_array_ident,
                                    nf.createExpressionStatement(_yield_expr));

        // iterator
        auto iterName = MLIRHelper::getAnonymousName(location, ".iter", getFullNamespaceName());

        NodeArray<Statement> statements;
        statements.push_back(forOfStat);
        auto block = nf.createBlock(statements, false);
        auto funcIter =
            nf.createFunctionExpression(undefined, nf.createToken(SyntaxKind::AsteriskToken),
                                        nf.createIdentifier(convertUTF8toWide(iterName)), undefined, undefined, undefined, block);

        funcIter->pos.pos = pos;
        funcIter->_end = _end;

        // call
        NodeArray<Expression> emptyArguments;
        auto callOfIter = nf.createCallExpression(funcIter, undefined, emptyArguments);

        return mlirGen(callOfIter, genContext);
    }

    ValueOrLogicalResult mlirGenArrayFilter(mlir::Location location, ArrayRef<mlir::Value> operands,
                                            const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto arraySrc = operands[0];
        auto funcSrc = operands[1];

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto funcVarDecl = std::make_shared<VariableDeclarationDOM>(".func", funcSrc.getType(), location);
        DECLARE(funcVarDecl, funcSrc);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _func_ident = nf.createIdentifier(S(".func"));

        auto _v_ident = nf.createIdentifier(S(".v"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto [pos, _end] = LocationHelper::getSpan(location);

        auto _yield_expr = nf.createYieldExpression(undefined, _v_ident);
        _yield_expr->pos.pos = pos;
        _yield_expr->_end = _end;

        auto forOfStat = nf.createForOfStatement(
            undefined, declList, _src_array_ident,
            nf.createIfStatement(nf.createCallExpression(_func_ident, undefined, argumentsArray),
                                 nf.createExpressionStatement(_yield_expr),
                                 undefined));

        // iterator
        auto iterName = MLIRHelper::getAnonymousName(location, ".iter", getFullNamespaceName());

        NodeArray<Statement> statements;
        statements.push_back(forOfStat);
        auto block = nf.createBlock(statements, false);
        auto funcIter =
            nf.createFunctionExpression(undefined, nf.createToken(SyntaxKind::AsteriskToken),
                                        nf.createIdentifier(convertUTF8toWide(iterName)), undefined, undefined, undefined, block);
        funcIter->pos.pos = pos;
        funcIter->_end = _end;

        // call
        NodeArray<Expression> emptyArguments;
        auto callOfIter = nf.createCallExpression(funcIter, undefined, emptyArguments);

        return mlirGen(callOfIter, genContext);
    }

    ValueOrLogicalResult mlirGenArrayReduce(mlir::Location location, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___array_reduce";

        if (!existGenericFunctionMap(funcName))
        {
            auto src = S("function __array_reduce<T, R>(arr: T[], f: (s: R, v: T) => R, init: R) \
            {   \
                let r = init;   \
                for (const v of arr) r = f(r, v);   \
                return r;   \
            }");

            {
                MLIRLocationGuard vgLoc(overwriteLoc); 
                overwriteLoc = location;
                if (mlir::failed(parsePartialStatements(src)))
                {
                    assert(false);
                    return mlir::failure();
                }
            }
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        return mlirGenCallExpression(location, funcResult, {}, operands, genContext);
    }

    ValueOrLogicalResult mlirGenCallBuiltInFunction(
        mlir::Location location, mlir::Value actualFuncRefValue, NodeArray<TypeNode> typeArguments, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        // TODO: when you resolve names such as "print", "parseInt" should return names in mlirGen(Identifier)
        auto calleeName = actualFuncRefValue.getDefiningOp()->getAttrOfType<mlir::FlatSymbolRefAttr>(StringRef(IDENTIFIER_ATTR_NAME));
        auto functionName = calleeName.getValue();

        if (auto thisSymbolRefOp = actualFuncRefValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            // do not remove it, it is needed for custom methods to be called correctly
            operands.insert(operands.begin(), thisSymbolRefOp.getThisVal());
        }

        // temp hack
        if (functionName == "__array_foreach")
        {
            mlirGenArrayForEach(location, operands, genContext);
            return mlir::success();
        }

        if (functionName == "__array_every")
        {
            return mlirGenArrayEvery(location, operands, genContext);
        }

        if (functionName == "__array_some")
        {
            return mlirGenArraySome(location, operands, genContext);
        }

        if (functionName == "__array_map")
        {
            return mlirGenArrayMap(location, operands, genContext);
        }

        if (functionName == "__array_filter")
        {
            return mlirGenArrayFilter(location, operands, genContext);
        }

        if (functionName == "__array_reduce")
        {
            return mlirGenArrayReduce(location, operands, genContext);
        }

        // resolve function           
        MLIRCustomMethods cm(builder, location, compileOptions);
        mlir::SmallVector<mlir::Type> typeArgs;
        for (auto typeArgNode : typeArguments)
        {
            auto typeArg = getType(typeArgNode, genContext);
            if (!typeArg)
            {
                return mlir::failure();
            }

            typeArgs.push_back(typeArg);
        }

        return cm.callMethod(
            functionName, 
            typeArgs,
            operands, 
            std::bind(&MLIRGenImpl::cast, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), 
            genContext);
    }

    ValueOrLogicalResult mlirGenCallExpression(mlir::Location location, mlir::Value funcResult,
                                               NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                               const GenContext &genContext)
    {
        GenContext specGenContext(genContext);
        specGenContext.callOperands = operands;

        // get function ref.
        auto result = mlirGenSpecialized(location, funcResult, typeArguments, operands, specGenContext);
        EXIT_IF_FAILED(result)
        auto actualFuncRefValue = V(result);

        if (!result.value && genContext.allowPartialResolve)
        {
            return mlir::success();
        }

        // special case when TypePredicateType is used in generic function and failed constraints 
        if (auto symbolRefOp = actualFuncRefValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            if (symbolRefOp.getIdentifier() == "")
            {
                if (auto funcType = mlir::dyn_cast<mlir_ts::FunctionType>(symbolRefOp.getType()))
                {
                    if (funcType.getNumInputs() == 0 && funcType.getNumResults() == 1)
                    {
                        if (auto litType = dyn_cast<mlir_ts::LiteralType>(funcType.getResult(0)))
                        {
                            return V(builder.create<mlir_ts::ConstantOp>(location, litType, litType.getValue()));                            
                        }
                    }
                }
            }
        }

        if (mth.isBuiltinFunctionType(actualFuncRefValue))
        {
            return mlirGenCallBuiltInFunction(location, 
                actualFuncRefValue, typeArguments, operands, genContext);
        }

        if (auto optFuncRef = dyn_cast<mlir_ts::OptionalType>(actualFuncRefValue.getType()))
        {
            CAST_A(condValue, location, getBooleanType(), actualFuncRefValue, genContext);

            auto resultType = mth.getReturnTypeFromFuncRef(optFuncRef.getElementType());

            LLVM_DEBUG(llvm::dbgs() << "\n!! Conditional call, return type: " << resultType << "\n";);

            auto hasReturn = !mth.isNoneType(resultType) && resultType != getVoidType();
            auto ifOp = hasReturn
                            ? builder.create<mlir_ts::IfOp>(location, getOptionalType(resultType), condValue, true)
                            : builder.create<mlir_ts::IfOp>(location, condValue, false);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

            // value if true

            auto innerFuncRef =
                builder.create<mlir_ts::ValueOp>(location, optFuncRef.getElementType(), actualFuncRefValue);

            auto result = mlirGenCallExpression(location, innerFuncRef, typeArguments, operands, genContext);
            auto value = V(result);
            if (value)
            {
                auto optValue =
                    builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});

                // else
                builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, getOptionalType(resultType));
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});
            }

            builder.setInsertionPointAfter(ifOp);

            if (hasReturn)
            {
                return ifOp.getResults().front();
            }

            return mlir::success();
        }

        return mlirGenCall(location, actualFuncRefValue, operands, genContext);
    }

    ValueOrLogicalResult NewClassInstanceOnStack(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        // seems we are calling type constructor
        // TODO: review it, really u should forbid to use "a = Class1();" to allocate in stack, or finish it
        // using Class..new(true) method

        return NewClassInstance(location, classType, operands, genContext, true /*on stack*/);
    }

    ValueOrLogicalResult NewClassInstance(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext, bool onStack = false)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        auto newOp = onStack 
            ? NewClassInstanceLogicAsOp(location, classType, onStack, genContext)
            : ValueOrLogicalResult(NewClassInstanceAsMethodCallOp(location, classInfo, true, genContext));
        EXIT_IF_FAILED_OR_NO_VALUE(newOp)
        if (mlir::failed(mlirGenCallConstructor(location, classInfo, V(newOp), operands, false, genContext)))
        {
            return mlir::failure();
        }

        return V(newOp);
    }

    ValueOrLogicalResult mlirGenCall(mlir::Location location, mlir::Value funcRefValue,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        ValueOrLogicalResult value(mlir::failure());
        mlir::TypeSwitch<mlir::Type>(funcRefValue.getType())
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, genContext);
            })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, genContext);
            })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledBoundFuncType) {
                auto calledFuncType =
                    getFunctionType(calledBoundFuncType.getInputs(), calledBoundFuncType.getResults(), calledBoundFuncType.isVarArg());
                auto thisValue = builder.create<mlir_ts::GetThisOp>(location, calledFuncType.getInput(0), funcRefValue);
                auto unboundFuncRefValue = builder.create<mlir_ts::GetMethodOp>(location, calledFuncType, funcRefValue);
                value = mlirGenCallFunction(location, calledFuncType, unboundFuncRefValue, thisValue, operands, genContext);
            })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledExtentFuncType) {
                auto calledFuncType =
                    getFunctionType(calledExtentFuncType.getInputs(), calledExtentFuncType.getResults(), calledExtentFuncType.isVarArg());
                if (auto createExtensionFunctionOp = funcRefValue.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
                {
                    auto thisValue = createExtensionFunctionOp.getThisVal();
                    auto funcRefValue = createExtensionFunctionOp.getFunc();
                    value = mlirGenCallFunction(location, calledFuncType, funcRefValue, thisValue, operands, genContext);
                }
                else
                {
                    emitError(location, "not supported");
                    value = mlir::Value();
                }
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                value = NewClassInstanceOnStack(location, classType, operands, genContext);
            })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                MLIRCodeLogic mcl(builder);
                auto refValue = mcl.GetReferenceFromValue(location, funcRefValue);
                if (refValue)
                {
                    // seems we are calling type constructor for super()
                    auto classInfo = getClassInfoByFullName(classStorageType.getName().getValue());
                    // to track result call
                    value = mlirGenCallConstructor(location, classInfo, refValue, operands, true, genContext);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Default([&](auto type) {
                // TODO: this is hack, rewrite it
                // it is not function, so just return value as maybe it has been resolved earlier like in case
                // "<number>.ToString()"
                value = funcRefValue;
            });

        return value;
    }

    template <typename T = mlir_ts::FunctionType>
    ValueOrLogicalResult mlirGenCallFunction(mlir::Location location, T calledFuncType, mlir::Value funcRefValue,
                                             SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        return mlirGenCallFunction(location, calledFuncType, funcRefValue, mlir::Value(), operands, genContext);
    }

    template <typename T = mlir_ts::FunctionType>
    ValueOrLogicalResult mlirGenCallFunction(mlir::Location location, T calledFuncType, mlir::Value funcRefValue,
                                             mlir::Value thisValue, SmallVector<mlir::Value, 4> &operands,
                                             const GenContext &genContext)
    {
        if (thisValue)
        {
            operands.insert(operands.begin(), thisValue);
        }

        if (mlir::failed(mlirGenPrepareCallOperands(location, operands, calledFuncType.getInputs(), calledFuncType.isVarArg(),
                                             genContext)))
        {
            return mlir::failure();
        }
        else
        {
            for (auto &oper : operands)
            {
                VALIDATE(oper, location)
            }

            // if last is vararg
            auto isNativeVarArgsCall = false;
            if (calledFuncType.isVarArg())
            {
                auto varArgsType = calledFuncType.getInputs().back();
                auto fromIndex = calledFuncType.getInputs().size() - 1;
                auto toIndex = operands.size();

                LLVM_DEBUG(llvm::dbgs() << "\n!! isVarArg type (array), type: " << varArgsType << "\n";);
                LLVM_DEBUG(llvm::dbgs() << "\t last value = " << operands.back() << "\n";);

                // check if vararg is prepared earlier
                auto isVarArgPreparedAlready = (toIndex - fromIndex) == 1 && (operands.back().getType() == varArgsType)
                  || isNativeVarArgsCall;
                if (!isVarArgPreparedAlready)
                {
                    SmallVector<mlir::Value, 4> varArgOperands;
                    for (auto i = fromIndex; i < toIndex; i++)
                    {
                        varArgOperands.push_back(operands[i]);
                    }

                    operands.pop_back_n(toIndex - fromIndex);

                    // create array
                    auto array = varArgOperands.empty() && !isa<mlir_ts::ArrayType>(varArgsType)
                        ? V(builder.create<mlir_ts::UndefOp>(location, varArgsType))
                        : V(builder.create<mlir_ts::CreateArrayOp>(location, varArgsType, varArgOperands));
                    operands.push_back(array);

                    LLVM_DEBUG(for (auto& ops : varArgOperands) llvm::dbgs() << "\t value = " << ops << "\n";);
                }
            }

            VALIDATE_FUNC(calledFuncType, location)

            // default
            auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(
                MLIRHelper::getCallSiteLocation(funcRefValue, location),
                funcRefValue, operands);

            if (calledFuncType.getResults().size() > 0)
            {
                auto callValue = callIndirectOp.getResult(0);
                auto hasReturn = callValue.getType() != getVoidType();
                if (hasReturn)
                {
                    return callValue;
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenPrepareCallOperands(mlir::Location location, SmallVector<mlir::Value, 4> &operands,
                                            mlir::ArrayRef<mlir::Type> argFuncTypes, bool isVarArg,
                                            const GenContext &genContext)
    {
        int opArgsCount = operands.size();
        int funcArgsCount = argFuncTypes.size();

        if (mlir::failed(mlirGenAdjustOperandTypes(location, operands, argFuncTypes, isVarArg, genContext)))
        {
            return mlir::failure();
        }

        if (funcArgsCount > opArgsCount)
        {
            auto lastArgIndex = argFuncTypes.size() - 1;

            // -1 to exclude count params
            for (auto i = (size_t)opArgsCount; i < funcArgsCount; i++)
            {
                if (i == 0)
                {
                    if (auto refType = dyn_cast<mlir_ts::RefType>(argFuncTypes[i]))
                    {
                        if (isa<mlir_ts::TupleType>(refType.getElementType()))
                        {
                            llvm_unreachable("capture or this ref is not resolved.");
                            return mlir::failure();
                        }
                    }
                }

                if (isVarArg && i >= lastArgIndex)
                {
                    break;
                }

                operands.push_back(builder.create<mlir_ts::UndefOp>(location, argFuncTypes[i]));
            }
        }

        return mlir::success();
    }

    struct OperandsProcessingInfo
    {
        OperandsProcessingInfo(mlir::Type funcType, SmallVector<mlir::Value, 4> &operands, int offsetArgs, bool noReceiverTypesForGenericCall, MLIRTypeHelper &mth, bool disableSpreadParam) 
            : operands{operands}, lastArgIndex{-1}, hasType{false}, hasVarArgs{false}, currentParameter{offsetArgs}, noReceiverTypesForGenericCall{noReceiverTypesForGenericCall}, mth{mth}
        {
            detectVarArgTypeInfo(funcType, disableSpreadParam);
        }

        void detectVarArgTypeInfo(mlir::Type funcType, bool disableSpreadParam)
        {
            auto tupleParamsType = mth.getParamsTupleTypeFromFuncRef(funcType);
            if (!tupleParamsType || isa<mlir::NoneType>(tupleParamsType))
            {
                return;
            }

            hasType = true;
            parameters = mlir::cast<mlir_ts::TupleType>(tupleParamsType).getFields();
            lastArgIndex = parameters.size() - 1;
            if (!disableSpreadParam && mth.getVarArgFromFuncRef(funcType))
            {
                hasVarArgs = true;
                varArgType = parameters.back().type;
                // unwrap array type to get elementType
                if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(varArgType))
                {
                    varArgType = arrayType.getElementType();
                    if (mth.isGenericType(varArgType))
                    {
                        // in case of generics which are not defined yet, array will be identified later in generic method call
                        varArgType = mlir::Type();
                        hasVarArgs = false;
                    }
                }
                else
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! VarArg type is: " << varArgType << "\n";);
                    // in case of generics which are not defined yet, array will be identified later in generic method call
                    varArgType = mlir::Type();
                    hasVarArgs = false;
                }
            }
        }

        mlir::Type getReceiverType()
        {
            if (!hasType)
            {
                return mlir::Type();
            }

            if (isVarArg() && currentParameter >= lastArgIndex)
            {
                return varArgType;
            }

            auto receiverType = 
                currentParameter < parameters.size() 
                    ? parameters[currentParameter].type 
                    : mlir::Type();
            return receiverType;
        }

        void setReceiverTo(GenContext &argGenContext)
        {
            if (!hasType)
            {
                return;
            }

            argGenContext.receiverFuncType = getReceiverType();
            argGenContext.receiverType = 
                !noReceiverTypesForGenericCall 
                    ? argGenContext.receiverFuncType 
                    : mlir::Type();
        }

        mlir::Type isCastNeededWithOptionalUnwrap(mlir::Type type)
        {
            return isCastNeeded(type, true);
        }

        mlir::Type isCastNeeded(mlir::Type type, bool isOptionalUnwrap = false)
        {
            auto receiverType = getReceiverType();
            if (isOptionalUnwrap) 
            {
                receiverType = mth.stripOptionalType(receiverType);
            }

            return receiverType && type != receiverType 
                ? receiverType 
                : mlir::Type();
        }

        void nextParameter()
        {
            ++currentParameter;
        }

        bool isVarArg() 
        {
            return currentParameter == lastArgIndex && hasVarArgs;
        }

        auto restCount()
        {
            return lastArgIndex - currentParameter + 1;
        }

        void addOperand(mlir::Value value)
        {
            operands.push_back(value);
        }

        void addOperandAndMoveToNextParameter(mlir::Value value)
        {
            addOperand(value);
            nextParameter();
        }

        SmallVector<mlir::Value, 4> &operands;
        llvm::ArrayRef<mlir::typescript::FieldInfo> parameters;
        int lastArgIndex;
        mlir::Type varArgType;
        bool hasType;
        bool hasVarArgs;
        int currentParameter;
        bool noReceiverTypesForGenericCall;
        MLIRTypeHelper &mth;
    };

    mlir::LogicalResult processOperandSpreadElement(mlir::Location location, mlir::Value source, OperandsProcessingInfo &operandsProcessingInfo, const GenContext &genContext)
    {
        auto count = operandsProcessingInfo.restCount();

        auto nextPropertyType = evaluateProperty(location, source, ITERATOR_NEXT, genContext);
        if (nextPropertyType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadElement, next type is: " << nextPropertyType << "\n";);

            auto returnType = mth.getReturnTypeFromFuncRef(nextPropertyType);
            if (returnType)
            {
                // as tuple or const_tuple
                ::llvm::ArrayRef<mlir_ts::FieldInfo> fields;
                mlir::TypeSwitch<mlir::Type>(returnType)
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { fields = tupleType.getFields(); })
                    .template Case<mlir_ts::ConstTupleType>(
                        [&](auto constTupleType) { fields = constTupleType.getFields(); })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                auto propValue = mlir::StringAttr::get(builder.getContext(), "value");
                if (std::any_of(fields.begin(), fields.end(), [&] (auto field) { return field.id == propValue; }))
                {
                    // treat it as <???>.next().value structure
                    // property
                    auto nextProperty = mlirGenPropertyAccessExpression(location, source, ITERATOR_NEXT, false, genContext);

                    for (auto spreadIndex = 0;  spreadIndex < count; spreadIndex++)
                    {
                        // call nextProperty
                        SmallVector<mlir::Value, 4> callOperands;
                        auto callResult = mlirGenCall(location, nextProperty, callOperands, genContext);
                        EXIT_IF_FAILED_OR_NO_VALUE(callResult)

                        // load property "value"
                        auto doneProperty = mlirGenPropertyAccessExpression(location, callResult, "done", false, genContext);
                        EXIT_IF_FAILED_OR_NO_VALUE(doneProperty)

                        auto valueProperty = mlirGenPropertyAccessExpression(location, callResult, "value", false, genContext);
                        EXIT_IF_FAILED_OR_NO_VALUE(valueProperty)

                        auto valueProp = V(valueProperty);

                        if (auto receiverType = operandsProcessingInfo.isCastNeededWithOptionalUnwrap(valueProp.getType()))
                        {
                            CAST(valueProp, location, receiverType, valueProp, genContext);
                        }                        

                        // conditional expr:  done ? undefined : value
                        auto doneInvValue =  V(builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(),
                            builder.getI32IntegerAttr((int)SyntaxKind::ExclamationToken), doneProperty));

                        mlir::Value condValue;
                        // if (isa<mlir_ts::AnyType>(valueProp.getType()))
                        // {
                        //     condValue = anyOrUndefined(location, doneInvValue, [&](auto genContext) { return valueProp; }, genContext);
                        // }
                        // else
                        // {
                            condValue = builder.create<mlir_ts::OptionalOp>(location, getOptionalType(valueProp.getType()), valueProp, doneInvValue);
                        // }

                        operandsProcessingInfo.addOperandAndMoveToNextParameter(condValue);
                    }
                }
                else
                {
                    llvm_unreachable("not implemented");
                }

                return mlir::success();    
            }
        }                                        

        if (auto lengthPropertyType = evaluateProperty(location, source, LENGTH_FIELD_NAME, genContext))
        {
            // treat it as <???>[index] structure
            auto lengthProperty = mlirGenPropertyAccessExpression(location, source, LENGTH_FIELD_NAME, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(lengthProperty)

            auto elementType = evaluateElementAccess(location, source, false, genContext);
            if (genContext.receiverType && genContext.receiverType != elementType)
            {
                elementType = genContext.receiverType;
            }

            auto valueFactory =
            (isa<mlir_ts::AnyType>(elementType))
                ? std::bind(&MLIRGenImpl::anyOrUndefined, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)
                : std::bind(&MLIRGenImpl::optionalValueOrUndefined, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

            for (auto spreadIndex = 0;  spreadIndex < count; spreadIndex++)
            {
                auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getStructIndexType(),
                                                    mth.getStructIndexAttrValue(spreadIndex));

                // conditional expr:  length > "spreadIndex" ? value[index] : undefined
                auto inBoundsValue =  V(builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                    builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanToken), 
                    lengthProperty,
                    indexVal));

                auto spreadValue = valueFactory(location, inBoundsValue, 
                    [&](auto genContext) { 
                        auto result = mlirGenElementAccess(location, source, indexVal, false, genContext); 
                        EXIT_IF_FAILED_OR_NO_VALUE(result)
                        auto value = V(result);

                        if (auto receiverType = operandsProcessingInfo.isCastNeeded(value.getType()))
                        {
                            CAST(value, location, receiverType, value, genContext);
                        }

                        return ValueOrLogicalResult(value);
                    }, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(spreadValue)

                operandsProcessingInfo.addOperandAndMoveToNextParameter(spreadValue);
            }

            return mlir::success();
        }

        // this is defualt behavior for tuple
        // treat it as <???>[index] structure
        for (auto spreadIndex = 0;  spreadIndex < count; spreadIndex++)
        {
            auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getStructIndexType(),
                                                mth.getStructIndexAttrValue(spreadIndex));

            auto result = mlirGenElementAccess(location, source, indexVal, false, genContext); 
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto value = V(result);

            operandsProcessingInfo.addOperandAndMoveToNextParameter(value);
        }

        return mlir::success();        
    }

    mlir::LogicalResult mlirGenOperand(Expression expression, OperandsProcessingInfo &operandsProcessingInfo, const GenContext &genContext)
    {
        GenContext argGenContext(genContext);
        argGenContext.clearReceiverTypes();
        operandsProcessingInfo.setReceiverTo(argGenContext);

        auto result = mlirGen(expression, argGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto value = V(result);

        if (expression == SyntaxKind::SpreadElement)
        {
            auto location = loc(expression);
            if (mlir::failed(processOperandSpreadElement(location, value, operandsProcessingInfo, argGenContext)))
            {
                return mlir::failure();
            }

            return mlir::success();
        }

        operandsProcessingInfo.addOperandAndMoveToNextParameter(value);
        return mlir::success();
    }

    mlir::LogicalResult mlirGenOperandVarArgs(mlir::Location location, int processedArgs, NodeArray<Expression> arguments, 
        OperandsProcessingInfo &operandsProcessingInfo, const GenContext &genContext)
    {
        // calculate array context
        SmallVector<ArrayElement> values;
        struct ArrayInfo arrayInfo{};

        // set receiver type
        auto elementReceiverType = operandsProcessingInfo.getReceiverType();
        if (elementReceiverType)
        {
            auto receiverType = mlir_ts::ArrayType::get(elementReceiverType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! varargs - receiver type: " << receiverType << "\n";);
            // TODO: isGenericType is applied as hack here, find out the issue
            // I think it should be operandsProcessingInfo.noReceiverTypesForGenericCall in setReceiver
            arrayInfo.setReceiver(receiverType, 
                operandsProcessingInfo.noReceiverTypesForGenericCall || mth.isGenericType(genContext.receiverType));
        }

        for (auto it = arguments.begin() + processedArgs; it != arguments.end(); ++it)
        {
            if (mlir::failed(processArrayElementForValues(*it, values, arrayInfo, genContext)))
            {
                return mlir::failure();
            }
        }

        arrayInfo.adjustArrayType(getAnyType());

        auto varArgOperandValue = createArrayFromArrayInfo(location, values, arrayInfo, genContext);
        operandsProcessingInfo.addOperand(varArgOperandValue);

        return mlir::success();
    }

    // TODO: rewrite code (do as clean as ArrayLiteral)
    mlir::LogicalResult mlirGenOperands(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands,
                                        mlir::Type funcType, const GenContext &genContext, int offsetArgs = 0, bool noReceiverTypesForGenericCall = false)
    {
        OperandsProcessingInfo operandsProcessingInfo(funcType, operands, offsetArgs, noReceiverTypesForGenericCall, mth, genContext.disableSpreadParams);

        for (auto it = arguments.begin(); it != arguments.end(); ++it)
        {
            if (operandsProcessingInfo.isVarArg())
            {
                auto proccessedArgs = std::distance(arguments.begin(), it);
                return mlirGenOperandVarArgs(loc(arguments), proccessedArgs, arguments, operandsProcessingInfo, genContext);
            }            

            if (mlir::failed(mlirGenOperand(*it, operandsProcessingInfo, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenAdjustOperandTypes(mlir::Location location, SmallVector<mlir::Value, 4> &operands,
                                                  mlir::ArrayRef<mlir::Type> argFuncTypes, bool isVarArg,
                                                  const GenContext &genContext)
    {
        auto i = 0; // we need to shift in case of 'this'
        auto lastArgIndex = argFuncTypes.size() - 1;
        mlir::Type varArgType;
        if (isVarArg)
        {
            auto lastType = argFuncTypes.back();
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(lastType))
            {
                lastType = arrayType.getElementType();
            }

            varArgType = lastType;
        }

        for (auto value : operands)
        {
            VALIDATE(value, location)

            mlir::Type argTypeDestFuncType = {};
            if (i >= argFuncTypes.size() && !isVarArg)
            {
                // emitError(location)
                //     << "function does not have enough parameters to accept all arguments, arg #" << i;
                // return mlir::failure();

                // to support native variadic calls
                break;
            }

            if (isVarArg && i >= lastArgIndex)
            {
                argTypeDestFuncType = varArgType;

                // if we have processed VarArg - do nothing
                if (i == lastArgIndex 
                    && lastArgIndex == operands.size() - 1
                    && value.getType() == getArrayType(varArgType))
                {
                    // nothing todo 
                    break;
                }
            }
            else
            {
                argTypeDestFuncType = argFuncTypes[i];
            }

            if (value.getType() != argTypeDestFuncType)
            {
                CAST_A(castValue, location, argTypeDestFuncType, value, genContext);
                operands[i] = castValue;
            }

            i++;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenSetVTableToInstance(mlir::Location location, ClassInfo::TypePtr classInfo,
                                                   mlir::Value thisValue, const GenContext &genContext)
    {
        auto virtualTable = classInfo->getHasVirtualTable();
        if (!virtualTable)
        {
            return mlir::success();
        }

        auto result = mlirGenPropertyAccessExpression(location, thisValue, VTABLE_NAME, genContext);
        auto vtableVal = V(result);
        MLIRCodeLogic mcl(builder);
        auto vtableRefVal = mcl.GetReferenceFromValue(location, vtableVal);

        // vtable symbol reference
        auto fullClassVTableFieldName = concat(classInfo->fullName, VTABLE_NAME);
        auto vtableAddress = resolveFullNameIdentifier(location, fullClassVTableFieldName, true, genContext);

        mlir::Value vtableValue;
        if (vtableAddress)
        {
            CAST_A(castedValue, location, getOpaqueType(), vtableAddress, genContext);
            vtableValue = castedValue;
        }
        else
        {
            // we need to calculate VTable type
            /*
            llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
            classInfo->getVirtualTable(virtualTable);
            auto virtTuple = getVirtualTableType(virtualTable);

            auto classVTableRefOp = builder.create<mlir_ts::AddressOfOp>(
                location, mlir_ts::RefType::get(virtTuple), fullClassVTableFieldName, ::mlir::IntegerAttr());

            CAST_A(castedValue, location, getOpaqueType(), classVTableRefOp, genContext);
            vtableValue = castedValue;
            */

            // vtable type will be detected later
            auto classVTableRefOp = builder.create<mlir_ts::AddressOfOp>(
                location, getOpaqueType(), fullClassVTableFieldName, ::mlir::IntegerAttr());

            vtableValue = classVTableRefOp;
        }

        builder.create<mlir_ts::StoreOp>(location, vtableValue, vtableRefVal);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, ClassInfo::TypePtr classInfo,
                                               mlir::Value thisValue, SmallVector<mlir::Value, 4> &operands,
                                               bool castThisValueToClass, const GenContext &genContext)
    {
        assert(classInfo);

        auto virtualTable = classInfo->getHasVirtualTable();
        auto hasConstructor = classInfo->getHasConstructor();
        if (!hasConstructor && !virtualTable)
        {
            return mlir::success();
        }

        auto effectiveThisValue = thisValue;
        if (castThisValueToClass)
        {
            CAST(effectiveThisValue, location, classInfo->classType, thisValue, genContext);
        }

        if (classInfo->getHasConstructor())
        {
            auto propAccess =
                mlirGenPropertyAccessExpression(location, effectiveThisValue, CONSTRUCTOR_NAME, false, genContext);

            if (!propAccess && !genContext.allowPartialResolve)
            {
                emitError(location) << "Call Constructor: can't find constructor";
            }

            EXIT_IF_FAILED_OR_NO_VALUE(propAccess)
            return mlirGenCall(location, propAccess, operands, genContext);
        }

        return mlir::success();
    }

    // TODO: refactor it, somehow when NewClassInstanceAsMethodCallOp calling Ctor and NewClassInstanceLogicAsOp is not
    ValueOrLogicalResult NewClassInstance(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
                                          NodeArray<TypeNode> typeArguments, bool suppressConstructorCall, 
                                          const GenContext &genContext)
    {

        auto type = value.getType();
        type = mth.convertConstTupleTypeToTupleType(type);

        assert(type);

        auto resultType = type;
        if (mth.isValueType(type))
        {
            resultType = getValueRefType(type);
        }

        // if true, will call Class..new method, otheriwise ts::NewOp which we need to implement Class..new method 
        auto methodCallWay = !suppressConstructorCall;

        mlir::Value newOp;
        if (auto classType = dyn_cast<mlir_ts::ClassType>(resultType))
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            if (!classInfo)
            {
                auto genericClassInfo = getGenericClassInfoByFullName(classType.getName().getValue());
                if (genericClassInfo)
                {
                    emitError(location) << "Generic class '"<< to_print(classType) << "' is missing type arguments ";
                    return mlir::failure(); 
                }

                emitError(location) << "Can't find class " << to_print(classType);
                return mlir::failure(); 
            }

            if (genContext.dummyRun)
            {
                // just to cut a lot of calls
                newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(false));
                return newOp;
            }

            auto newOp = NewClassInstanceAsMethodCallOp(location, classInfo, methodCallWay, genContext);
            if (!newOp)
            {
                return mlir::failure();
            }

            if (methodCallWay)
            {
                // evaluate constructor
                mlir::Type tupleParamsType;
                auto funcValueRef = evaluateProperty(location, newOp, CONSTRUCTOR_NAME, genContext);
                if (funcValueRef)
                {
                    SmallVector<mlir::Value, 4> operands;
                    if (mlir::failed(mlirGenOperands(arguments, operands, funcValueRef, genContext, 1/*this params shift*/)))
                    {
                        emitError(location) << "Call constructor: can't resolve values of all parameters";
                        return mlir::failure();
                    }

                    assert(newOp);
                    auto result  = mlirGenCallConstructor(location, classInfo, newOp, operands, false, genContext);
                    EXIT_IF_FAILED(result)
                }
            }

            return newOp;
        }

        return NewClassInstanceLogicAsOp(location, resultType, false, genContext);
    }

    ValueOrLogicalResult NewClassInstanceLogicAsOp(mlir::Location location, mlir::Type typeOfInstance, bool stackAlloc,
                                                   const GenContext &genContext)
    {
        if (auto classType = dyn_cast<mlir_ts::ClassType>(typeOfInstance))
        {
            // set virtual table
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            if (!classInfo)
            {
                auto genericClassInfo = getGenericClassInfoByFullName(classType.getName().getValue());
                if (genericClassInfo)
                {
                    emitError(location) << "Generic class '"<< to_print(classType) << "' is missing type arguments ";
                    return mlir::failure(); 
                }

                emitError(location) << "Can't find class " << to_print(classType);
                return mlir::Value(); 
            }

            return NewClassInstanceLogicAsOp(location, classInfo, stackAlloc, genContext);
        }

        auto newOp = builder.create<mlir_ts::NewOp>(location, typeOfInstance, builder.getBoolAttr(stackAlloc));
        return V(newOp);
    }

    mlir::Value NewClassInstanceLogicAsOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool stackAlloc,
                                          const GenContext &genContext)
    {
        mlir::Value newOp;
#if ENABLE_TYPED_GC
        auto enabledGC = !compileOptions.disableGC;
        if (enabledGC && !stackAlloc)
        {
            auto typeDescrType = builder.getI64Type();
            auto typeDescGlobalName = getTypeDescriptorFieldName(classInfo);
            auto typeDescRef = resolveFullNameIdentifier(location, typeDescGlobalName, true, genContext);
            auto typeDescCurrentValue = builder.create<mlir_ts::LoadOp>(location, typeDescrType, typeDescRef);

            CAST_A(condVal, location, getBooleanType(), typeDescCurrentValue, genContext);

            auto ifOp = builder.create<mlir_ts::IfOp>(
                location, mlir::TypeRange{typeDescrType}, condVal,
                [&](mlir::OpBuilder &opBuilder, mlir::Location loc) {
                    builder.create<mlir_ts::ResultOp>(loc, mlir::ValueRange{typeDescCurrentValue});
                },
                [&](mlir::OpBuilder &opBuilder, mlir::Location loc) {
                    // call typr bitmap
                    auto fullClassStaticFieldName = getTypeBitmapMethodName(classInfo);

                    auto funcType = getFunctionType({}, {typeDescrType}, false);

                    auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                        location, funcType,
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), fullClassStaticFieldName));

                    auto callIndirectOp =
                        builder.create<mlir_ts::CallIndirectOp>(
                            MLIRHelper::getCallSiteLocation(funcSymbolOp->getLoc(), location),
                            funcSymbolOp, mlir::ValueRange{});
                    auto typeDescr = callIndirectOp.getResult(0);

                    // save value
                    builder.create<mlir_ts::StoreOp>(location, typeDescr, typeDescRef);

                    builder.create<mlir_ts::ResultOp>(loc, mlir::ValueRange{typeDescr});
                });

            auto typeDescrValue = ifOp.getResult(0);

            assert(!stackAlloc);
            newOp = builder.create<mlir_ts::GCNewExplicitlyTypedOp>(location, classInfo->classType, typeDescrValue);
        }
        else
        {
            newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(stackAlloc));
        }
#else
        newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(stackAlloc));
#endif
        mlirGenSetVTableToInstance(location, classInfo, newOp, genContext);
        return newOp;
    }

    mlir::Value NewClassInstanceAsMethodCallOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool asMethodCall,
                                             const GenContext &genContext)
    {
#ifdef USE_NEW_AS_METHOD
        if (asMethodCall)
        {
            auto classRefVal = builder.create<mlir_ts::ClassRefOp>(
                location, classInfo->classType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));

            // call <Class>..new to create new instance
            auto result = mlirGenPropertyAccessExpression(location, classRefVal, NEW_METHOD_NAME, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto newFuncRef = V(result);

            assert(newFuncRef);

            SmallVector<mlir::Value, 4> emptyOperands;
            auto resultCall = mlirGenCallExpression(location, newFuncRef, {}, emptyOperands, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(resultCall)
            auto newOp = V(resultCall);
            return newOp;
        }
#endif

        return NewClassInstanceLogicAsOp(location, classInfo, false, genContext);
    }

    ValueOrLogicalResult NewArray(mlir::Location location, mlir::Type type, NodeArray<Expression> arguments, const GenContext &genContext)
    {
        mlir::Type elementType;
        if (auto arrayType = dyn_cast_or_null<mlir_ts::ArrayType>(type))
        {
            elementType = arrayType.getElementType();
        }

        if (!elementType)
        {
            return mlir::failure();
        }

        elementType = mth.convertConstTupleTypeToTupleType(elementType);

        mlir::Value count;
        if (arguments.size() == 0)
        {
            count = builder.create<mlir_ts::ConstantOp>(location, builder.getIntegerType(32, false), builder.getUI32IntegerAttr(0));
        }
        else if (arguments.size() == 1)
        {
            auto result = mlirGen(arguments.front(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            count = V(result);           
        }
        else
        {
            SmallVector<ArrayElement> values;
            struct ArrayInfo arrayInfo{};

            GenContext noReceiverGenContext(genContext);
            noReceiverGenContext.clearReceiverTypes();
            noReceiverGenContext.receiverType = mlir::cast<mlir_ts::ArrayType>(getArrayType(elementType));

            if (mlir::failed(processArrayValues(arguments, values, arrayInfo, noReceiverGenContext)))
            {
                return mlir::failure();
            }

            return createArrayFromArrayInfo(location, values, arrayInfo, genContext);
        }

        if (count.getType() != builder.getI32Type())
        {
            // TODO: test cast result
            count = cast(location, builder.getI32Type(), count, genContext);
        }

        auto newArrOp = builder.create<mlir_ts::NewArrayOp>(location, getArrayType(elementType), count);
        return V(newArrOp);                     
    }    

    ValueOrLogicalResult NewClassInstanceByCallingNewCtor(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
            NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto result = mlirGenPropertyAccessExpression(location, value, NEW_CTOR_METHOD_NAME, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto newCtorMethod = V(result);        

        SmallVector<mlir::Value, 4> operands;
        if (mlir::failed(mlirGenOperands(arguments, operands, newCtorMethod.getType(), genContext)))
        {
            emitError(location) << "Call new instance: can't resolve values of all parameters";
            return mlir::failure();
        }

        return mlirGenCallExpression(location, newCtorMethod, typeArguments, operands, genContext);        
    }

    ValueOrLogicalResult mlirGen(NewExpression newExpression, const GenContext &genContext)
    {
        auto location = loc(newExpression);

        // 3 cases, name, index access, method call
        mlir::Type type;
        auto typeExpression = newExpression->expression;
        ////auto isNewArray = typeExpression == SyntaxKind::ElementAccessExpression && newExpression->arguments.isTextRangeEmpty();
        auto result = mlirGen(typeExpression, newExpression->typeArguments, genContext);
        if (result.failed())
        {
            if (typeExpression == SyntaxKind::Identifier)
            {
                // TODO: review it, seems it should be resolved earlier
                auto name = MLIRHelper::getName(typeExpression.as<Identifier>());
                type = findEmbeddedType(location, name, newExpression->typeArguments, genContext);
                if (type)
                {
                    result = V(builder.create<mlir_ts::TypeRefOp>(location, type));
                }
            }
        }

        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto value = V(result);

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(value.getType()))
        {
            return NewArray(location, arrayType, newExpression->arguments, genContext);
        }

        // to support custom Array<T>
        if (auto classType = dyn_cast<mlir_ts::ClassType>(value.getType()))
        {
            if (newExpression->typeArguments > 0 && classType.getName().getValue().starts_with("Array<"))
            {
                auto arrayType = findEmbeddedType(location, "Array", newExpression->typeArguments, genContext);
                if (arrayType)
                {
                    return NewArray(location, arrayType, newExpression->arguments, genContext);
                }
            }
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(value.getType()))
        {
            return NewClassInstanceByCallingNewCtor(location, value, newExpression->arguments, newExpression->typeArguments, genContext);
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(value.getType()))
        {
            auto newCtorMethod = evaluateProperty(location, value, NEW_CTOR_METHOD_NAME, genContext);
            if (newCtorMethod)
            {
                return NewClassInstanceByCallingNewCtor(location, value, newExpression->arguments, newExpression->typeArguments, genContext);
            }
        }

        // default - class instance
        auto suppressConstructorCall = (newExpression->internalFlags & InternalFlags::SuppressConstructorCall) ==
                                        InternalFlags::SuppressConstructorCall;

        return NewClassInstance(location, value, newExpression->arguments, newExpression->typeArguments, suppressConstructorCall, genContext);
    }

    mlir::LogicalResult mlirGen(DeleteExpression deleteExpression, const GenContext &genContext)
    {

        auto location = loc(deleteExpression);

        auto result = mlirGen(deleteExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expr = V(result);

        if (!isa<mlir_ts::RefType>(expr.getType()) && !isa<mlir_ts::ValueRefType>(expr.getType()) &&
            !isa<mlir_ts::ClassType>(expr.getType()))
        {
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(expr.getType()))
            {
                CAST(expr, location, mlir_ts::RefType::get(arrayType.getElementType()), expr, genContext);
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        builder.create<mlir_ts::DeleteOp>(location, expr);

        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(VoidExpression voidExpression, const GenContext &genContext)
    {

        auto location = loc(voidExpression);

        auto result = mlirGen(voidExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expr = V(result);

        auto value = getUndefined(location);

        return value;
    }

    ValueOrLogicalResult mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext)
    {
        auto location = loc(typeOfExpression);

        auto result = mlirGen(typeOfExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto resultValue = V(result);
        // auto typeOfValue = builder.create<mlir_ts::TypeOfOp>(location, getStringType(), resultValue);
        // return V(typeOfValue);

        // needed to use optimizers
        TypeOfOpHelper toh(builder);
        auto typeOfValue = toh.typeOfLogic(location, resultValue, resultValue.getType());
        return typeOfValue;
    }

    ValueOrLogicalResult mlirGen(NonNullExpression nonNullExpression, const GenContext &genContext)
    {
        return mlirGen(nonNullExpression->expression, genContext);
    }

    ValueOrLogicalResult mlirGen(OmittedExpression ommitedExpression, const GenContext &genContext)
    {
        auto location = loc(ommitedExpression);

        return V(builder.create<mlir_ts::UndefOp>(location, getUndefinedType()));
    }

    ValueOrLogicalResult mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(templateExpressionAST);

        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto text = convertWideToUTF8(templateExpressionAST->head->rawText);
        auto head = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(text));

        // first string
        strs.push_back(head);
        for (auto span : templateExpressionAST->templateSpans)
        {
            auto expression = span->expression;
            auto result = mlirGen(expression, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto exprValue = V(result);

            if (exprValue.getType() != stringType)
            {
                CAST(exprValue, location, stringType, exprValue, genContext);
            }

            // expr value
            strs.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            // text
            strs.push_back(spanValue);
        }

        if (strs.size() <= 1)
        {
            return V(head);
        }

        auto concatValues =
            builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return V(concatValues);
    }

    ValueOrLogicalResult mlirGen(TaggedTemplateExpression taggedTemplateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(taggedTemplateExpressionAST);

        auto templateExpressionAST = taggedTemplateExpressionAST->_template;

        SmallVector<mlir::Attribute, 4> strs;
        SmallVector<mlir::Value, 4> vals;

        std::string text = convertWideToUTF8(
            templateExpressionAST->head 
                ? templateExpressionAST->head->rawText 
                : templateExpressionAST->rawText);

        // first string
        strs.push_back(getStringAttr(text));
        for (auto span : templateExpressionAST->templateSpans)
        {
            // expr value
            auto expression = span->expression;
            auto result = mlirGen(expression, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto exprValue = V(result);

            vals.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            // text
            strs.push_back(getStringAttr(spanText));
        }

        // tag method
        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), strs);
        auto constStringArray =
            builder.create<mlir_ts::ConstantOp>(location, getConstArrayType(getStringType(), strs.size()), arrayAttr);

        CAST_A(strArrayValue, location, getArrayType(getStringType()), constStringArray, genContext);

        vals.insert(vals.begin(), strArrayValue);

        auto result = mlirGen(taggedTemplateExpressionAST->tag, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto callee = V(result);

        if (!mth.isAnyFunctionType(callee.getType()))
        {
            emitError(location, "is not callable");
            return mlir::failure();
        }

        VALIDATE_FUNC(callee.getType(), location)

        auto inputs = mth.getParamsFromFuncRef(callee.getType());

        SmallVector<mlir::Value, 4> operands;

        auto i = 0;
        for (auto value : vals)
        {
            if (inputs.size() <= i)
            {
                emitError(location, "not matching to tag parameters count");
                return mlir::Value();
            }

            if (value.getType() != inputs[i])
            {
                CAST_A(castValue, location, inputs[i], value, genContext);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            i++;
        }

        // call
        auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(
            MLIRHelper::getCallSiteLocation(callee, location),
            callee, operands);
        if (callIndirectOp.getNumResults() > 0)
        {
            return callIndirectOp.getResult(0);
        }

        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(NullLiteral nullLiteral, const GenContext &genContext)
    {
        return V(builder.create<mlir_ts::NullOp>(loc(nullLiteral), getNullType()));
    }

    mlir_ts::LiteralType getBooleanLiteral(bool val) 
    {
        auto attrVal = mlir::BoolAttr::get(builder.getContext(), val);
        auto literalType = mlir_ts::LiteralType::get(attrVal, getBooleanType());
        return literalType;
    }

    ValueOrLogicalResult mlirGenBooleanValue(mlir::Location location, bool val)
    {
        auto literalType = getBooleanLiteral(val);
        return V(builder.create<mlir_ts::ConstantOp>(location, literalType, literalType.getValue()));
    }

    ValueOrLogicalResult mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
    {
        return mlirGenBooleanValue(loc(trueLiteral), true);
    }

    ValueOrLogicalResult mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
    {
        return mlirGenBooleanValue(loc(falseLiteral), false);
    }

    mlir::Attribute getIntTypeAttribute(string text)
    {
        APSInt newVal(wstos(text));

        auto unsignedVal = false;
        auto width = newVal.getBitWidth();
        switch (width)
        {
        //case 8:
        //case 16:
        case 32:
        case 64:
        case 128:
            unsignedVal = true;
            break;
        default:
            //if (width < 8) width = 8; else 
            //if (width < 16) width = 16; else 
            if (width < 32) width = 32; else 
            if (width < 64) width = 64; else 
            if (width < 128) width = 128;
            else llvm_unreachable("not implemented");
        }

        auto type = builder.getIntegerType(width, !unsignedVal);
        return mlir::IntegerAttr::get(type, newVal.getExtValue());
    }

    mlir::Attribute getNumericLiteralAttribute(NumericLiteral numericLiteral)
    {
        if (numericLiteral->text.find_first_of(S(".eE")) == string::npos)
        {
            return getIntTypeAttribute(numericLiteral->text);
        }

#ifdef NUMBER_F64        
        auto f64 = builder.getF64Type();
        llvm::APFloat val(f64.getFloatSemantics(), wstos(numericLiteral->text.c_str()));
        return builder.getFloatAttr(f64, val.convertToDouble());
#else
        auto f32 = builder.getF32Type();
        llvm::APFloat val(f32.getFloatSemantics(), wstos(numericLiteral->text.c_str()));
        return builder.getFloatAttr(f32, val.convertToFloat());
#endif
    }

    ValueOrLogicalResult mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
    {
        auto attrVal = getNumericLiteralAttribute(numericLiteral);
        auto attrType = mlir::cast<mlir::TypedAttr>(attrVal).getType();
        auto valueType = isa<mlir::FloatType>(attrType) ? getNumberType() : attrType;
        auto literalType = mlir_ts::LiteralType::get(attrVal, valueType);
        return V(builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext)
    {
        APSInt newVal(wstos(
            *(bigIntLiteral->text.end() - 1) == S('n') 
                ? bigIntLiteral->text.substr(0, bigIntLiteral->text.length() - 1) 
                : bigIntLiteral->text.c_str()));
        auto type = builder.getI64Type();
        auto attrVal = mlir::IntegerAttr::get(type, newVal.getExtValue());
        auto literalType = mlir_ts::LiteralType::get(attrVal, type);
        return V(builder.create<mlir_ts::ConstantOp>(loc(bigIntLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult mlirGenStringValue(mlir::Location location, StringRef text, bool asString = false)
    {
        auto attrVal = getStringAttr(text);
        auto literalType = asString ? (mlir::Type)getStringType() : (mlir::Type)mlir_ts::LiteralType::get(attrVal, getStringType());
        return V(builder.create<mlir_ts::ConstantOp>(location, literalType, attrVal));
    }

    ValueOrLogicalResult mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
    {
        auto text = convertWideToUTF8(stringLiteral->text);
        return mlirGenStringValue(loc(stringLiteral), text);
    }

    ValueOrLogicalResult mlirGen(ts::RegularExpressionLiteral regularExpressionLiteral, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        auto regName = nf.createIdentifier(S("RegExp"));

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(
            nf.createStringLiteral(
                regularExpressionLiteral->text, 
                false, 
                regularExpressionLiteral->hasExtendedUnicodeEscape));

        auto newRegExpr = nf.createNewExpression(regName, undefined, argumentsArray);

        LLVM_DEBUG(printDebug(newRegExpr););

        return mlirGen(newRegExpr, genContext);
    }

    ValueOrLogicalResult mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral,
                                 const GenContext &genContext)
    {
        auto text = convertWideToUTF8(noSubstitutionTemplateLiteral->text);

        auto attrVal = getStringAttr(text);
        auto literalType = mlir_ts::LiteralType::get(attrVal, getStringType());
        return V(builder.create<mlir_ts::ConstantOp>(loc(noSubstitutionTemplateLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult mlirGenAppendArrayByEachElement(mlir::Location location, mlir::Value arrayDest, mlir::Value arraySrc,
                                            const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", arraySrc.getType(), location);
        DECLARE(srcArrayVarDecl, arraySrc);

        auto dstArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".dst_array", arrayDest.getType(), location);
        dstArrayVarDecl->setReadWriteAccess(true);
        DECLARE(dstArrayVarDecl, arrayDest);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _dst_array_ident = nf.createIdentifier(S(".dst_array"));

        auto _push_ident = nf.createIdentifier(S("push"));

        auto _v_ident = nf.createIdentifier(S(".v"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        // access to push
        auto pushExpr = nf.createPropertyAccessExpression(_dst_array_ident, _push_ident);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto forOfStat = nf.createForOfStatement(
            undefined, declList, _src_array_ident,
            nf.createExpressionStatement(nf.createCallExpression(pushExpr, undefined, argumentsArray)));

        LLVM_DEBUG(printDebug(forOfStat););

        return mlirGen(forOfStat, genContext);
    }

    enum class TypeData
    {
        NotSet,
        Array,
        Tuple
    };

    struct RecevierContext
    {
        RecevierContext() : receiverTupleTypeIndex{-1} {}

        void set(mlir_ts::ArrayType arrayType)
        {
            receiverElementType = arrayType.getElementType();
        }

        void set(mlir_ts::TupleType tupleType)
        {
            receiverTupleType = tupleType;
        }

        void setReceiverTo(GenContext &noReceiverGenContext)
        {        
            noReceiverGenContext.receiverType = (receiverElementType) ? receiverElementType : mlir::Type();
        }

        mlir::Type isCastNeeded(mlir::Type type)
        {
            return receiverElementType && type != receiverElementType 
                ? receiverElementType 
                : mlir::Type();
        }

        void nextTupleField()
        {
            if (!receiverTupleType)
            {
                return;
            }

            receiverElementType =
                receiverTupleType.size() > ++receiverTupleTypeIndex
                ? receiverTupleType.getFieldInfo(receiverTupleTypeIndex).type 
                : mlir::Type();
        }

        mlir::Type receiverElementType;
        mlir_ts::TupleType receiverTupleType;
        int receiverTupleTypeIndex;
    };

    struct ArrayInfo
    {
        ArrayInfo() : recevierContext(), 
            dataType{TypeData::NotSet},
            isConst{true},
            anySpreadElement{false},
            applyCast{false},
            notAllItemsTheSameType{false}
        {
        }

        void set(mlir_ts::ArrayType arrayType, bool isReceiverGenericType)
        {
            dataType = TypeData::Array;
            arrayElementType = arrayType.getElementType();
            if (!isReceiverGenericType)
                accumulatedArrayElementType = 
                    arrayElementType;
        }        

        void setReceiverArray(mlir_ts::ArrayType arrayType, bool isReceiverGenericType)
        {        
            set(arrayType, isReceiverGenericType);
            recevierContext.set(arrayType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! array elements - receiver type: " << recevierContext.receiverElementType << "\n";);
        }

        // TODO: check code if tupleType is genericType and merge is correct
        void set(mlir_ts::TupleType tupleType)
        {
            dataType = TypeData::Tuple;
            arrayElementType = tupleType;
        }

        void setReceiverTuple(mlir_ts::TupleType tupleType)
        {  
            set(tupleType);
            recevierContext.set(tupleType);
        }        

        // TODO: review all receivers in case of generic types in generic functions, to avoid merging T with actual types
        void setReceiver(mlir::Type type, bool isReceiverGenericType)
        {
            MLIRTypeHelper mth(nullptr);
            type = mth.stripOptionalType(type);

            mlir::TypeSwitch<mlir::Type>(type)
                .template Case<mlir_ts::ArrayType>([&](auto a) { isReceiverGenericType ? set(a, isReceiverGenericType) : setReceiverArray(a, isReceiverGenericType); })
                .template Case<mlir_ts::TupleType>([&](auto t) { isReceiverGenericType ? set(t) : setReceiverTuple(t); })
                .Default([&](auto type) {
                    // just ignore it
                });
        }        

        void adjustArrayType(mlir::Type defaultElementType)
        {
            // post processing values
            if (anySpreadElement || dataType == TypeData::NotSet)
            {
                // this is array
                dataType = TypeData::Array;
            }

            if (dataType == TypeData::Tuple 
                && (recevierContext.receiverTupleType == mlir::Type()) 
                && !isa<mlir_ts::UnionType>(accumulatedArrayElementType))
            {
                // seems we can convert tuple into array, for example [1.0, 2, 3] -> [1.0, 2.0, 3.0]
                dataType = TypeData::Array;
                applyCast = true;
            }

            if (dataType == TypeData::Array)
            {
                arrayElementType = 
                    accumulatedArrayElementType 
                        ? accumulatedArrayElementType 
                        : defaultElementType;

                if (recevierContext.receiverElementType && recevierContext.receiverElementType != arrayElementType)
                {
                    arrayElementType = recevierContext.receiverElementType;
                    applyCast = true;
                }

                if (notAllItemsTheSameType)
                {
                    applyCast = true;
                }
            }
        }

        RecevierContext recevierContext;

        TypeData dataType;
        mlir::Type accumulatedArrayElementType;
        mlir::Type arrayElementType;
        bool isConst;
        bool anySpreadElement;
        bool applyCast;
        bool notAllItemsTheSameType;
    };

    struct ArrayElement
    {
    public:
        mlir::Value value;
        bool isSpread;
        bool isVariableSizeOfSpreadElement;
    };

    mlir::LogicalResult accumulateArrayItemType(mlir::Location location, mlir::Type type, struct ArrayInfo &arrayInfo) 
    {
        auto elementType = arrayInfo.accumulatedArrayElementType;

        // TODO: special case (should we use [] = as const_array<undefined, 0> instead of const_array<any, 0>?)
        if (auto constArray = dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            if (constArray.getSize() == 0)
            {
                return mlir::success();
            }
        }

        // if we have receiver type we do not need to "adopt it"
        auto wideType = arrayInfo.recevierContext.receiverElementType ? type : mth.wideStorageType(type);

        //LLVM_DEBUG(llvm::dbgs() << "\n!! element type: " << wideType << " original type: " << type << "\n";);

        elementType = elementType ? elementType : wideType;
        if (elementType != wideType)
        {
            if (arrayInfo.dataType == TypeData::NotSet)
            {
                // presumably it is tuple
                arrayInfo.dataType = TypeData::Tuple;
            }
            
            auto merged = false;
            elementType = mth.mergeType(location, elementType, wideType, merged);
        }

        //LLVM_DEBUG(llvm::dbgs() << "\n!! result element type: " << elementType << "\n";);

        arrayInfo.accumulatedArrayElementType = elementType;

        arrayInfo.notAllItemsTheSameType |= arrayInfo.accumulatedArrayElementType != wideType;

        return mlir::success();
    };

    mlir::LogicalResult processArrayValuesSpreadElement(mlir::Location location, mlir::Value itemValue, SmallVector<ArrayElement> &values, struct ArrayInfo &arrayInfo, const GenContext &genContext)
    {
        arrayInfo.anySpreadElement = true;
        arrayInfo.isConst = false;

        auto type = itemValue.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadElement, src type: " << type << "\n";);

        if (auto constArray = dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            auto constantOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>();
            auto arrayAttr = mlir::cast<mlir::ArrayAttr>(constantOp.getValue());
            // TODO: improve it with using array concat
            for (auto [index, val] : enumerate(arrayAttr))
            {
                auto indexVal = builder.create<mlir_ts::ConstantOp>(location, builder.getIntegerType(32), builder.getI32IntegerAttr(index));
                auto result = mlirGenElementAccess(location, itemValue, indexVal, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result);
                auto newConstVal = V(result);
                values.push_back({newConstVal, false, false});
            }

            accumulateArrayItemType(location, constArray.getElementType(), arrayInfo);

            return mlir::success();
        }
        
        if (auto array = dyn_cast<mlir_ts::ArrayType>(type))
        {
            // TODO: implement method to concat array with const-length array in one operation without using 'push' for each element
            values.push_back({itemValue, true, true});

            auto arrayElementType = mth.wideStorageType(array.getElementType());
            accumulateArrayItemType(location, arrayElementType, arrayInfo);

            return mlir::success();
        }

        if (auto array = dyn_cast<mlir_ts::StringType>(type))
        {
            // TODO: implement method to concat array with const-length array in one operation without using 'push' for each element
            values.push_back({itemValue, true, true});

            accumulateArrayItemType(location, getCharType(), arrayInfo);

            return mlir::success();
        }        

        auto nextPropertyType = evaluateProperty(location, itemValue, ITERATOR_NEXT, genContext);
        if (nextPropertyType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadElement, next type is: " << nextPropertyType << "\n";);

            auto returnType = mth.getReturnTypeFromFuncRef(nextPropertyType);
            if (returnType)
            {
                // as tuple or const_tuple
                ::llvm::ArrayRef<mlir_ts::FieldInfo> fields;
                mlir::TypeSwitch<mlir::Type>(returnType)
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { fields = tupleType.getFields(); })
                    .template Case<mlir_ts::ConstTupleType>(
                        [&](auto constTupleType) { fields = constTupleType.getFields(); })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                auto propValue = mlir::StringAttr::get(builder.getContext(), "value");
                if (std::any_of(fields.begin(), fields.end(), [&] (auto field) { return field.id == propValue; }))
                {
                    arrayInfo.isConst = false;

                    values.push_back({itemValue, true, true});

                    auto arrayElementType = mth.wideStorageType(fields.front().type);
                    accumulateArrayItemType(location, arrayElementType, arrayInfo);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }

                return mlir::success();    
            }
        }                                        

        // DO NOT PUT before xxx.next() property otherwise ""..."" for Iterator will not work
        if (auto constTuple = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            // because it is tuple it may not have the same types
            arrayInfo.isConst = false;

            if (auto constantOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto arrayAttr = mlir::cast<mlir::ArrayAttr>(constantOp.getValue());
                auto index = -1;
                for (auto val : arrayAttr)
                {
                    MLIRPropertyAccessCodeLogic cl(builder, location, itemValue, builder.getIndexAttr(++index));
                    auto newConstVal = cl.Tuple(constTuple, true);

                    values.push_back({newConstVal, false, false});

                    accumulateArrayItemType(location, constTuple.getFieldInfo(index).type, arrayInfo);
                }
            }

            return mlir::success();                
        }       
        
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            values.push_back({itemValue, true, false});
            for (auto tupleItem : tupleType)
            {
                accumulateArrayItemType(location, tupleItem.type, arrayInfo);
            }

            return mlir::success();
        }                           

        LLVM_DEBUG(llvm::dbgs() << "\n!! spread element type: " << type << "\n";);

        return mlir::success();
    }
    
    mlir::LogicalResult processArrayElementForValues(Expression item, SmallVector<ArrayElement> &values, struct ArrayInfo &arrayInfo, const GenContext &genContext)
    {
        auto location = loc(item);

        auto &recevierContext = arrayInfo.recevierContext;

        recevierContext.nextTupleField();

        GenContext noReceiverGenContext(genContext);
        noReceiverGenContext.clearReceiverTypes();
        recevierContext.setReceiverTo(noReceiverGenContext);

        auto result = mlirGen(item, noReceiverGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto itemValue = V(result);
        if (itemValue.getDefiningOp<mlir_ts::UndefOp>())
        {
            // process ommited expression
            if (auto optionalType = dyn_cast_or_null<mlir_ts::OptionalType>(recevierContext.receiverElementType))
            {
                itemValue = builder.create<mlir_ts::OptionalUndefOp>(location, recevierContext.receiverElementType);
            }
        }

        auto type = itemValue.getType();

        if (item == SyntaxKind::SpreadElement)
        {
            if (mlir::failed(processArrayValuesSpreadElement(location, itemValue, values, arrayInfo, genContext)))
            {
                return mlir::failure();
            }
        }
        else
        {
            if (auto castType = recevierContext.isCastNeeded(type))
            {
                CAST(itemValue, location, castType, itemValue, genContext);
                type = itemValue.getType();
            }

            if (!itemValue.getDefiningOp<mlir_ts::ConstantOp>() || 
            // TODO: in case of [{ a: '', b: 0, c: '' }, { a: "", b: 3, c: 0 }]
                ((arrayInfo.dataType == TypeData::Array || arrayInfo.dataType == TypeData::NotSet)
                    && isa<mlir_ts::ConstTupleType>(itemValue.getType())                 
                    && arrayInfo.accumulatedArrayElementType 
                    && mth.removeConstType(itemValue.getType()) != arrayInfo.accumulatedArrayElementType))
            {
                arrayInfo.isConst = false;
            }                

            values.push_back({itemValue, false, false});
            accumulateArrayItemType(location, type, arrayInfo);
        }

        return mlir::success();
    }

    mlir::LogicalResult processTupleTailingOptionalValues(mlir::Location location, int processedValues, SmallVector<ArrayElement> &values, struct ArrayInfo &arrayInfo, const GenContext &genContext)
    {
        if (!arrayInfo.recevierContext.receiverTupleType)
        {
            return mlir::success();
        }

        if (processedValues >= arrayInfo.recevierContext.receiverTupleType.getFields().size())
        {
            return mlir::success();
        }

        auto &recevierContext = arrayInfo.recevierContext;
        for (auto i = processedValues; i < arrayInfo.recevierContext.receiverTupleType.getFields().size(); i++)
        {
            recevierContext.nextTupleField();
            if (!isa<mlir_ts::OptionalType>(recevierContext.receiverElementType))
            {
                emitError(location, "value is not provided for non-optional type");
                return mlir::failure();
            }

            auto undefVal = builder.create<mlir_ts::OptionalUndefOp>(location, recevierContext.receiverElementType);
            values.push_back({undefVal, false, false});
        }

        return mlir::success();
    }    

    mlir::LogicalResult processArrayValues(NodeArray<Expression> arrayElements, SmallVector<ArrayElement> &values, struct ArrayInfo &arrayInfo, const GenContext &genContext)
    {
        // check receiverType
        if (genContext.receiverType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! array/tuple - receiver type: " << genContext.receiverType << "\n";);
            // TODO: isGenericType is applied as hack here, find out the issue
            arrayInfo.setReceiver(genContext.receiverType, mth.isGenericType(genContext.receiverType));
        }

        for (auto &item : arrayElements)
        {
            if (mlir::failed(processArrayElementForValues(item, values, arrayInfo, genContext)))
            {
                return mlir::failure();
            }
        }

        if (mlir::failed(processTupleTailingOptionalValues(loc(arrayElements), arrayElements.size(), values, arrayInfo, genContext)))
        {
            return mlir::failure();
        }

        arrayInfo.adjustArrayType(getAnyType());

        return mlir::success();
    }

    ValueOrLogicalResult createConstArrayOrTuple(mlir::Location location, ArrayRef<ArrayElement> values, struct ArrayInfo arrayInfo, const GenContext &genContext)
    {
        // collect const values as attributes
        SmallVector<mlir::Attribute> constValues;
        for (auto &itemValue : values)
        {
            auto constOp = itemValue.value.getDefiningOp<mlir_ts::ConstantOp>();
            if (arrayInfo.applyCast)
            {
                constValues.push_back(mth.convertAttrIntoType(constOp.getValueAttr(), arrayInfo.arrayElementType, builder)); 
            }
            else
            {
                constValues.push_back(constOp.getValueAttr()); 
            }
        }

        SmallVector<mlir::Type> constTypes;
        for (auto &itemValue : values)
        {
            auto type = mth.wideStorageType(itemValue.value.getType());
            constTypes.push_back(type);
        }

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), constValues);
        if (arrayInfo.dataType == TypeData::Tuple)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;
            for (auto type : constTypes)
            {
                fieldInfos.push_back({mlir::Attribute(), type});
            }

            return V(
                builder.create<mlir_ts::ConstantOp>(location, getConstTupleType(fieldInfos), arrayAttr));
        }

        if (arrayInfo.dataType == TypeData::Array)
        {
            auto arrayElementType = arrayInfo.arrayElementType ? arrayInfo.arrayElementType : getAnyType();

            return V(builder.create<mlir_ts::ConstantOp>(
                location, getConstArrayType(arrayElementType, constValues.size()), arrayAttr));
        }

        llvm_unreachable("not implemented");
    }

    ValueOrLogicalResult createTupleFromArrayLiteral(mlir::Location location, ArrayRef<ArrayElement> values, struct ArrayInfo arrayInfo, const GenContext &genContext)
    {
        SmallVector<mlir::Value> arrayValues;
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        for (auto val : values)
        {
            fieldInfos.push_back({mlir::Attribute(), val.value.getType()});
            arrayValues.push_back(val.value);
        }

        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues));
    }    

    ValueOrLogicalResult createFixedSizeArrayFromArrayLiteral(mlir::Location location, ArrayRef<ArrayElement> values, struct ArrayInfo arrayInfo, const GenContext &genContext)
    {
        SmallVector<mlir::Value> arrayValues;
        for (auto val : values)
        {
            auto arrayValue = val.value;
            if (arrayInfo.applyCast)
            {
                CAST(arrayValue, location, arrayInfo.arrayElementType, val.value, genContext)
            }

            arrayValues.push_back(arrayValue);
        }

        auto newArrayOp =
            builder.create<mlir_ts::CreateArrayOp>(location, getArrayType(arrayInfo.arrayElementType), arrayValues);
        return V(newArrayOp);
    }

    ValueOrLogicalResult createDynamicArrayFromArrayLiteral(mlir::Location location, ArrayRef<ArrayElement> values, struct ArrayInfo arrayInfo, const GenContext &genContext)
    {
        MLIRCustomMethods cm(builder, location, compileOptions);
        SmallVector<mlir::Value> emptyArrayValues;
        auto arrType = getArrayType(arrayInfo.arrayElementType);
        auto newArrayOp = builder.create<mlir_ts::CreateArrayOp>(location, arrType, emptyArrayValues);
        auto varArray = builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(arrType),
                newArrayOp, builder.getBoolAttr(false), builder.getIndexAttr(0));

        auto loadedVarArray = builder.create<mlir_ts::LoadOp>(location, arrType, varArray);

        // TODO: push every element into array
        for (auto val : values)
        {
            if (val.isVariableSizeOfSpreadElement)
            {
                mlirGenAppendArrayByEachElement(location, varArray, val.value, genContext);
            }
            else
            {
                SmallVector<mlir::Value> vals;
                if (!val.isSpread)
                {
                    mlir::Value finalVal = val.value;
                    if (arrayInfo.arrayElementType != val.value.getType())
                    {
                        auto result = cast(location, arrayInfo.arrayElementType, val.value, genContext) ;
                        EXIT_IF_FAILED_OR_NO_VALUE(result)
                        finalVal = V(result);
                    }
                    else
                    {
                        finalVal = val.value;
                    }

                    vals.push_back(finalVal);
                }
                // to process const tuple & tuple
                else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(mth.convertConstTupleTypeToTupleType(val.value.getType())))
                {
                    llvm::SmallVector<mlir::Type> destTupleTypes;
                    if (mlir::succeeded(mth.getFieldTypes(tupleType, destTupleTypes)))
                    {
                        auto resValues = builder.create<mlir_ts::DeconstructTupleOp>(location, destTupleTypes, val.value);
                        for (auto tupleVal : resValues.getResults())
                        {
                            mlir::Value finalVal;
                            if (arrayInfo.arrayElementType != tupleVal.getType())
                            {
                                auto result = cast(location, arrayInfo.arrayElementType, tupleVal, genContext) ;
                                EXIT_IF_FAILED_OR_NO_VALUE(result)
                                finalVal = V(result);
                            }
                            else
                            {
                                finalVal = tupleVal;
                            }

                            vals.push_back(finalVal);
                        }
                    }
                    else
                    {
                        return mlir::failure();
                    }
                }
                else
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! array spread value type: " << val.value.getType() << "\n";);
                    llvm_unreachable("not implemented");
                }
                
                assert(vals.size() > 0);

                cm.mlirGenArrayPush(location, loadedVarArray, vals);
            }
        }

        auto loadedVarArray2 = builder.create<mlir_ts::LoadOp>(location, arrType, varArray);
        return V(loadedVarArray2);
    }

    ValueOrLogicalResult createArrayFromArrayInfo(mlir::Location location, ArrayRef<ArrayElement> values, struct ArrayInfo arrayInfo, const GenContext &genContext)
    {
        if (arrayInfo.isConst)
        {
            return createConstArrayOrTuple(location, values, arrayInfo, genContext);
        }

        if (arrayInfo.dataType == TypeData::Tuple)
        {
            return createTupleFromArrayLiteral(location, values, arrayInfo, genContext);
        }

        if (!arrayInfo.anySpreadElement)
        {
            return createFixedSizeArrayFromArrayLiteral(location, values, arrayInfo, genContext);
        }

        return createDynamicArrayFromArrayLiteral(location, values, arrayInfo, genContext);
    }

    ValueOrLogicalResult mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        SmallVector<ArrayElement> values;
        struct ArrayInfo arrayInfo{};
        if (mlir::failed(processArrayValues(arrayLiteral->elements, values, arrayInfo, genContext)))
        {
            return mlir::failure();
        }

        return createArrayFromArrayInfo(location, values, arrayInfo, genContext);
    }

    // TODO: replace usage of this method with getFields method
    mlir::Type getTypeByFieldNameFromReceiverType(mlir::Attribute fieldName, mlir::Type receiverType)
    {
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(receiverType))
        {
            auto index = tupleType.getIndex(fieldName);
            if (index >= 0)
            {
                return tupleType.getType(index);
            }
        }

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(receiverType))
        {
            auto index = constTupleType.getIndex(fieldName);
            if (index >= 0)
            {
                return constTupleType.getType(index);
            }
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(receiverType))
        {
            auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
            auto index = interfaceInfo->getFieldIndex(fieldName);
            if (index >= 0)
            {
                return interfaceInfo->fields[index].type;
            }
        }        

        return mlir::Type();
    }

    ValueOrLogicalResult mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext)
    {
        auto location = loc(objectLiteral);

        MLIRCodeLogic mcl(builder);

        // first value
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        SmallVector<size_t> methodInfos;
        SmallVector<std::pair<std::string, size_t>> methodInfosWithCaptures;
        SmallVector<std::pair<mlir::Attribute, mlir::Value>> fieldsToSet;

        auto receiverType = genContext.receiverType;
        if (receiverType)
        {
            receiverType = mth.stripOptionalType(receiverType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! Recevier type: " << receiverType << "\n";);

            if ((isa<mlir_ts::TupleType>(receiverType) || isa<mlir_ts::ConstTupleType>(receiverType) || isa<mlir_ts::InterfaceType>(receiverType))
                 && objectLiteral->properties.size() == 0)
            {
                // return undef tuple
                llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
                if (mlir::succeeded(mth.getFields(receiverType, destTupleFields)))
                {
                    auto tupleType = getTupleType(destTupleFields);
                    return V(builder.create<mlir_ts::UndefOp>(location, tupleType));
                }
            }
        }

        // Object This Type
        auto name = MLIRHelper::getAnonymousName(loc_check(objectLiteral), ".obj", getFullNamespaceName());
        auto objectNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
        auto objectStorageType = getObjectStorageType(objectNameSymbol);
        auto objThis = getObjectType(objectStorageType);
        
        auto addFuncFieldInfo = [&](mlir::Attribute fieldId, const std::string &funcName,
                                    mlir_ts::FunctionType funcType) {
            auto type = funcType;

            values.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
            fieldInfos.push_back({fieldId, type});

            if (getCaptureVarsMap().find(funcName) != getCaptureVarsMap().end())
            {
                methodInfosWithCaptures.push_back({funcName, fieldInfos.size() - 1});
            }
            else
            {
                methodInfos.push_back(fieldInfos.size() - 1);
            }
        };

        auto addFieldInfoToArrays = [&](mlir::Attribute fieldId, mlir::Type type) {
            values.push_back(builder.getUnitAttr());
            fieldInfos.push_back({fieldId, type});
        };

        auto addFieldInfo = [&](mlir::Attribute fieldId, mlir::Value itemValue, mlir::Type receiverElementType) {
            mlir::Type type;
            mlir::Attribute value;
            auto isConstValue = true;
            if (auto constOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>())
            {
                value = constOp.getValueAttr();
                type = constOp.getType();
            }
            else if (auto symRefOp = itemValue.getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                value = symRefOp.getIdentifierAttr();
                type = symRefOp.getType();
            }
            else if (auto undefOp = itemValue.getDefiningOp<mlir_ts::UndefOp>())
            {
                value = builder.getUnitAttr();
                type = undefOp.getType();
            }
            else
            {
                value = builder.getUnitAttr();
                type = itemValue.getType();
                isConstValue = false;
            }

            type = mth.wideStorageType(type);

            if (receiverElementType)
            {
                //LLVM_DEBUG(llvm::dbgs() << "\n!! Object field type and receiver type: " << type << " type: " << receiverElementType << "\n";);

                if (type != receiverElementType)
                {
                    value = builder.getUnitAttr();
                    itemValue = cast(location, receiverElementType, itemValue, genContext);
                    isConstValue = false;
                }

                type = receiverElementType;
                //LLVM_DEBUG(llvm::dbgs() << "\n!! Object field type (from receiver) - id: " << fieldId << " type: " << type << "\n";);
            }

            values.push_back(value);
            fieldInfos.push_back({fieldId, type});
            if (!isConstValue)
            {
                fieldsToSet.push_back({fieldId, itemValue});
            }
        };

        auto processFunctionLikeProto = [&](mlir::Attribute fieldId, FunctionLikeDeclarationBase &funcLikeDecl) {
            auto funcGenContext = GenContext(genContext);
            funcGenContext.clearScopeVars();
            funcGenContext.clearReceiverTypes();
            funcGenContext.thisType = objThis;

            funcLikeDecl->parent = objectLiteral;

            auto [funcOp, funcProto, result, isGeneric] = mlirGenFunctionPrototype(funcLikeDecl, funcGenContext);
            if (mlir::failed(result) || !funcOp)
            {
                return;
            }

            // fix this parameter type (taking in account that first type can be captured type)
            auto funcName = funcOp.getName().str();
            auto funcType = funcOp.getFunctionType();

            // process local vars in this context
            if (funcProto->getHasExtraFields())
            {
                // note: this code needed to store local variables for generators
                auto localVars = getLocalVarsInThisContextMap().find(funcName);
                if (localVars != getLocalVarsInThisContextMap().end())
                {
                    for (auto fieldInfo : localVars->getValue())
                    {
                        addFieldInfoToArrays(fieldInfo.id, fieldInfo.type);
                    }
                }
            }

            addFuncFieldInfo(fieldId, funcName, funcType);
        };

        auto processFunctionLike = [&](mlir_ts::ObjectType objThis, FunctionLikeDeclarationBase &funcLikeDecl) {
            auto funcGenContext = GenContext(genContext);
            funcGenContext.clearScopeVars();
            funcGenContext.clearReceiverTypes();
            funcGenContext.thisType = objThis;

            LLVM_DEBUG(llvm::dbgs() << "\n!! Object Process function with this type: " << objThis << "\n";);

            funcLikeDecl->parent = objectLiteral;

            mlir::OpBuilder::InsertionGuard guard(builder);
            mlirGenFunctionLikeDeclaration(funcLikeDecl, funcGenContext);
        };

        // add all fields
        for (auto &item : objectLiteral->properties)
        {
            mlir::Value itemValue;
            mlir::Attribute fieldId;
            mlir::Type receiverElementType;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer == SyntaxKind::FunctionExpression ||
                    propertyAssignment->initializer == SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                fieldId = TupleFieldName(propertyAssignment->name, genContext);

                if (receiverType)
                {
                    receiverElementType = getTypeByFieldNameFromReceiverType(fieldId, receiverType);
                }

                // TODO: send context with receiver type
                GenContext receiverTypeGenContext(genContext);
                receiverTypeGenContext.clearReceiverTypes();
                if (receiverElementType)
                {
                    receiverTypeGenContext.receiverType = receiverElementType;
                }

                auto result = mlirGen(propertyAssignment->initializer, receiverTypeGenContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                itemValue = V(result);

                // in case of Union type
                if (receiverType && !receiverElementType)
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! Detecting dest. union type with first field: " << fieldId << "\n";);

                    if (auto unionType = dyn_cast<mlir_ts::UnionType>(receiverType))
                    {
                        for (auto subType : unionType.getTypes())
                        {
                            auto possibleType = getTypeByFieldNameFromReceiverType(fieldId, subType);
                            if (possibleType == itemValue.getType())
                            {
                                LLVM_DEBUG(llvm::dbgs() << "\n!! we picked type from union: " << subType << "\n";);

                                receiverElementType = possibleType;
                                receiverType = subType;
                                break;
                            }
                        }
                    }
                }
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer == SyntaxKind::FunctionExpression ||
                    shorthandPropertyAssignment->initializer == SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto result = mlirGen(shorthandPropertyAssignment->name.as<Expression>(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                itemValue = V(result);

                fieldId = TupleFieldName(shorthandPropertyAssignment->name, genContext);
            }
            else if (item == SyntaxKind::MethodDeclaration || item == SyntaxKind::GetAccessor || item == SyntaxKind::SetAccessor)
            {
                continue;
            }
            else if (item == SyntaxKind::SpreadAssignment)
            {
                auto spreadAssignment = item.as<SpreadAssignment>();
                auto result = mlirGen(spreadAssignment->expression, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto tupleValue = V(result);

                LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadAssignment value: " << tupleValue << "\n";);

                auto tupleFields = [&] (::llvm::ArrayRef<mlir_ts::FieldInfo> fields) {
                    SmallVector<mlir::Type> types;
                    for (auto &field : fields)
                    {
                        types.push_back(field.type);
                    }

                    // deconstruct tuple
                    auto res = builder.create<mlir_ts::DeconstructTupleOp>(loc(spreadAssignment), types, tupleValue);

                    // read all fields
                    for (auto pair : llvm::zip(fields, res.getResults()))
                    {
                        addFieldInfo(
                            std::get<0>(pair).id, 
                            std::get<1>(pair), 
                            receiverType 
                                ? getTypeByFieldNameFromReceiverType(std::get<0>(pair).id, receiverType) 
                                : mlir::Type());
                    }
                };

                mlir::TypeSwitch<mlir::Type>(tupleValue.getType())
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { tupleFields(tupleType.getFields()); })
                    .template Case<mlir_ts::ConstTupleType>(
                        [&](auto constTupleType) { tupleFields(constTupleType.getFields()); })
                    .template Case<mlir_ts::InterfaceType>(
                        [&](auto interfaceType) { 
                            mlir::SmallVector<mlir_ts::FieldInfo> destFields;
                            if (mlir::succeeded(mth.getFields(interfaceType, destFields)))
                            {
                                if (auto srcInterfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue()))
                                {
                                    for (auto fieldInfo : destFields)
                                    {
                                        auto interfaceFieldInfo = srcInterfaceInfo->findField(fieldInfo.id);

                                        MLIRPropertyAccessCodeLogic cl(builder, location, tupleValue, fieldInfo.id);
                                        // TODO: implemenet conditional
                                        mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, tupleValue, interfaceFieldInfo->isConditional, cl, genContext); 
                                        addFieldInfo(fieldInfo.id, propertyAccess, receiverElementType);
                                    }
                                }
                            }
                        })
                    .template Case<mlir_ts::ClassType>(
                        [&](auto classType) { 
                            mlir::SmallVector<mlir_ts::FieldInfo> destFields;
                            if (mlir::succeeded(mth.getFields(classType, destFields)))
                            {
                                if (auto srcClassInfo = getClassInfoByFullName(classType.getName().getValue()))
                                {
                                    for (auto fieldInfo : destFields)
                                    {
                                        auto foundField = false;                                        
                                        auto classFieldInfo = srcClassInfo->findField(fieldInfo.id, foundField);

                                        MLIRPropertyAccessCodeLogic cl(builder, location, tupleValue, fieldInfo.id);
                                        // TODO: implemenet conditional
                                        mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, tupleValue, false, cl, genContext); 
                                        addFieldInfo(fieldInfo.id, propertyAccess, receiverElementType);
                                    }
                                }
                            }
                        })                        
                    .Default([&](auto type) { 
                        LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadAssignment not implemented for type: " << type << "\n";);
                        llvm_unreachable("not implemented"); 
                    });

                continue;
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1)");
            }

            assert(genContext.allowPartialResolve || itemValue);

            addFieldInfo(fieldId, itemValue, receiverElementType);
        }

        // update after processing all fields
        objectStorageType.setFields(fieldInfos);

        // process all methods
        for (auto &item : objectLiteral->properties)
        {
            mlir::Attribute fieldId;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    propertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = propertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                fieldId = TupleFieldName(propertyAssignment->name, genContext);
                processFunctionLikeProto(fieldId, funcLikeDecl);
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    shorthandPropertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = shorthandPropertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                fieldId = TupleFieldName(shorthandPropertyAssignment->name, genContext);
                processFunctionLikeProto(fieldId, funcLikeDecl);
            }
            else if (item == SyntaxKind::MethodDeclaration || item == SyntaxKind::GetAccessor || item == SyntaxKind::SetAccessor)
            {
                auto funcLikeDecl = item.as<FunctionLikeDeclarationBase>();
                fieldId = TupleFieldName(funcLikeDecl->name, genContext);

                if (item == SyntaxKind::GetAccessor || item == SyntaxKind::SetAccessor)
                {
                    auto stringVal = mlir::cast<mlir::StringAttr>(fieldId).getValue();
                    std::string newField;
                    raw_string_ostream rso(newField);
                    rso << (item == SyntaxKind::GetAccessor ? "get_" : "set_") << stringVal;
                    
                    fieldId = mlir::StringAttr::get(builder.getContext(), mlir::StringRef(newField).copy(stringAllocator));                    
                }

                processFunctionLikeProto(fieldId, funcLikeDecl);
            }          
        }

        // create accum. captures
        llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> accumulatedCaptureVars;

        for (auto &methodRefWithName : methodInfosWithCaptures)
        {
            auto funcName = std::get<0>(methodRefWithName);
            auto methodRef = std::get<1>(methodRefWithName);
            auto &methodInfo = fieldInfos[methodRef];

            if (auto funcType = dyn_cast<mlir_ts::FunctionType>(methodInfo.type))
            {
                auto captureVars = getCaptureVarsMap().find(funcName);
                if (captureVars != getCaptureVarsMap().end())
                {
                    // mlirGenResolveCapturedVars
                    for (auto &captureVar : captureVars->getValue())
                    {
                        if (accumulatedCaptureVars.count(captureVar.getKey()) > 0)
                        {
                            assert(accumulatedCaptureVars[captureVar.getKey()] == captureVar.getValue());
                        }

                        accumulatedCaptureVars[captureVar.getKey()] = captureVar.getValue();
                    }
                }
                else
                {
                    assert(false);
                }
            }
        }

        if (accumulatedCaptureVars.size() > 0)
        {
            // add all captured
            SmallVector<mlir::Value> accumulatedCapturedValues;
            if (mlir::failed(mlirGenResolveCapturedVars(location, accumulatedCaptureVars, accumulatedCapturedValues,
                                                        genContext)))
            {
                return mlir::failure();
            }

            auto capturedValue = mlirGenCreateCapture(location, mcl.CaptureType(accumulatedCaptureVars),
                                                      accumulatedCapturedValues, genContext);
            addFieldInfo(MLIRHelper::TupleFieldName(CAPTURED_NAME, builder.getContext()), capturedValue, mlir::Type());
        }

        // final type, update
        objectStorageType.setFields(fieldInfos);

        // process all methods
        for (auto &item : objectLiteral->properties)
        {
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    propertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = propertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    shorthandPropertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = shorthandPropertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
            else if (item == SyntaxKind::MethodDeclaration || item == SyntaxKind::GetAccessor || item == SyntaxKind::SetAccessor)
            {
                auto funcLikeDecl = item.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
        }

        auto constTupleTypeWithReplacedThis = getConstTupleType(fieldInfos);

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        auto constantVal =
            builder.create<mlir_ts::ConstantOp>(location, constTupleTypeWithReplacedThis, arrayAttr);
        if (fieldsToSet.empty())
        {
            //CAST_A(result, location, objectStorageType, constantVal, genContext);
            //return result;
            return V(constantVal);
        }

        auto tupleType = mth.convertConstTupleTypeToTupleType(constantVal.getType());
        auto tupleValue = mlirGenCreateTuple(location, tupleType, constantVal, fieldsToSet, genContext);
        //CAST_A(result, location, objectStorageType, tupleValue, genContext);
        //return result;
        return V(tupleValue);
    }

    ValueOrLogicalResult mlirGenCreateTuple(mlir::Location location, mlir::Type tupleType, mlir::Value initValue,
                                            SmallVector<std::pair<mlir::Attribute, mlir::Value>> &fieldsToSet,
                                            const GenContext &genContext)
    {
        // we need to cast it to tuple and set values
        auto tupleVar = builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(tupleType), initValue,
                                                            builder.getBoolAttr(false), builder.getIndexAttr(0));
        for (auto fieldToSet : fieldsToSet)
        {
            VALIDATE(fieldToSet.first, location)
            VALIDATE(fieldToSet.second, location)

            auto result = mlirGenPropertyAccessExpression(location, tupleVar, fieldToSet.first, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto getField = V(result);

            auto result2 = mlirGenSaveLogicOneItem(location, getField, fieldToSet.second, genContext);
            EXIT_IF_FAILED(result2)
            auto savedValue = V(result2);
        }

        auto loadedValue = builder.create<mlir_ts::LoadOp>(location, tupleType, tupleVar);
        return V(loadedValue);
    }

    ValueOrLogicalResult mlirGen(Identifier identifier, const GenContext &genContext)
    {
        auto location = loc(identifier);

        // resolve name
        auto name = MLIRHelper::getName(identifier);

        // info: can't validate it here, in case of "print" etc
        return mlirGen(location, name, genContext);
    }

    mlir::Value resolveIdentifierAsVariable(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (name.empty())
        {
            return mlir::Value();
        }

        auto value = symbolTable.lookup(name);
        if (value.second && value.first)
        {
            //LLVM_DEBUG(dbgs() << "\n!! resolveIdentifierAsVariable: " << name << " type: " << value.second->getType() <<  " value: " << value.first;);

            // begin of logic: outer vars
            auto valueRegion = value.first.getParentRegion();
            auto isOuterVar = false;
            // TODO: review code "valueRegion && valueRegion->getParentOp()" is to support async.execute
            if (genContext.funcOp && genContext.funcOp != tempFuncOp && valueRegion &&
                valueRegion->getParentOp() /* && valueRegion->getParentOp()->getParentOp()*/)
            {
                // auto funcRegion = const_cast<GenContext &>(genContext).funcOp.getCallableRegion();
                auto funcRegion = const_cast<GenContext &>(genContext).funcOp.getCallableRegion();

                isOuterVar = !funcRegion->isAncestor(valueRegion);
                // TODO: HACK
                if (isOuterVar && value.second->getIgnoreCapturing())
                {
                    // special case when "ForceConstRef" pointering to outer variable but it is not outer var
                    isOuterVar = false;
                }

                LLVM_DEBUG(if (isOuterVar) dbgs() << "\n!! outer var: [" << value.second->getName()
                                  << "] \n\n\tvalue region: " << *valueRegion->getParentOp()
                                  << " \n\n\tFuncOp: " << const_cast<GenContext &>(genContext).funcOp << "";);                
            }

            if (isOuterVar && genContext.passResult && !isGenericFunctionReference(value.first))
            {
                LLVM_DEBUG(dbgs() << "\n!! capturing var: [" << value.second->getName()
                                  << "] \n\tvalue pair: " << value.first << " \n\ttype: " << value.second->getType()
                                  << " \n\treadwrite: " << value.second->getReadWriteAccess() << "";);

                // valueRegion->viewGraph();
                // const_cast<GenContext &>(genContext).funcOpVarScope.getCallableRegion()->viewGraph();

                // special case, to prevent capturing ".a" because of reference to outer VaribleOp, which is hack (review
                // solution for it)
                genContext.passResult->outerVariables.insert({value.second->getName(), value.second});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            //LLVM_DEBUG(dbgs() << "\n!! variable: " << name << " type: " << value.first.getType() << "\n");

            // load value if memref
            auto valueType = mlir::cast<mlir_ts::RefType>(value.first.getType()).getElementType();
            return builder.create<mlir_ts::LoadOp>(location, valueType, value.first);
        }

        return mlir::Value();
    }

    mlir::LogicalResult mlirGenResolveCapturedVars(mlir::Location location,
                                                   llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> captureVars,
                                                   SmallVector<mlir::Value> &capturedValues,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        for (auto &item : captureVars)
        {
            auto result = mlirGen(location, item.first(), genContext);
            auto varValue = V(result);

            // review capturing by ref.  it should match storage type
            auto refValue = mcl.GetReferenceFromValue(location, varValue);
            if (refValue)
            {
                capturedValues.push_back(refValue);
                // set var as captures
                if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                {
                    varOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOp = refValue.getDefiningOp<mlir_ts::ParamOp>())
                {
                    paramOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOptOp = refValue.getDefiningOp<mlir_ts::ParamOptionalOp>())
                {
                    paramOptOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else
                {
                    // TODO: review it.
                    // find out if u need to ensure that data is captured and belong to VariableOp or ParamOp with
                    // captured = true
                    LLVM_DEBUG(llvm::dbgs()
                                   << "\n!! var must be captured when loaded from other Op: " << refValue << "\n";);
                    // llvm_unreachable("variable must be captured.");
                }
            }
            else
            {
                // this is not ref, this is const value
                capturedValues.push_back(varValue);
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult mlirGenCreateCapture(mlir::Location location, mlir::Type capturedType,
                                              SmallVector<mlir::Value> capturedValues, const GenContext &genContext)
    {
        LLVM_DEBUG(for (auto &val : capturedValues) llvm::dbgs() << "\n!! captured val: " << val << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! captured type: " << capturedType << "\n";);

        // add attributes to track which one sent by ref.
        auto captured = builder.create<mlir_ts::CaptureOp>(location, capturedType, capturedValues);
        return V(captured);
    }

    mlir::Value resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir_ts::FunctionType funcType,
                                           mlir::Value thisValue, bool addGenericAttrFlag,
                                           const GenContext &genContext)
    {
        // check if required capture of vars
        auto captureVars = getCaptureVarsMap().find(name);
        if (captureVars != getCaptureVarsMap().end())
        {
            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            if (addGenericAttrFlag)
            {
                funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! func with capture: first type: [ " << funcType.getInput(0)
                                    << " ], \n\tfunc name: " << name << " \n\tfunc type: " << funcType << "\n");

            SmallVector<mlir::Value> capturedValues;
            if (mlir::failed(mlirGenResolveCapturedVars(location, captureVars->getValue(), capturedValues, genContext)))
            {
                return mlir::Value();
            }

            MLIRCodeLogic mcl(builder);

            auto captureType = mcl.CaptureType(captureVars->getValue());
            auto result = mlirGenCreateCapture(location, captureType, capturedValues, genContext);
            auto captured = V(result);
            CAST_A(opaqueTypeValue, location, getOpaqueType(), captured, genContext);
            return builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType),
                                                                  opaqueTypeValue, funcSymbolOp);
        }

        if (thisValue)
        {
            auto thisFuncSymbolOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                location, getBoundFunctionType(funcType), thisValue, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            if (addGenericAttrFlag)
            {
                thisFuncSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            }

            return V(thisFuncSymbolOp);
        }

        auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
            location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
        if (addGenericAttrFlag)
        {
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
        }

        return V(funcSymbolOp);
    }

    mlir::Value resolveFunctionNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto funcOp = fn->getValue();
            auto funcType = funcOp.getFunctionType();
            auto funcName = funcOp.getName();

            return resolveFunctionWithCapture(location, funcName, funcType, mlir::Value(), false, genContext);
        }

        return mlir::Value();
    }

    mlir::Type resolveTypeByNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
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
            assert(typeAliasInfo);
            return typeAliasInfo;
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            if (!classInfo->classType)
            {
                emitError(location) << "can't find class: " << name << "\n";
                return mlir::Type();
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
                mlir::FlatSymbolRefAttr::get(builder.getContext(), getFullNamespaceName(name)), 
                enumTypeInfo.first, 
                enumTypeInfo.second);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto classInfo = getClassInfoByFullName(fullName);
            if (classInfo)
            {
                return classInfo->classType;
            }

            auto interfaceInfo = getInterfaceInfoByFullName(fullName);
            if (interfaceInfo)
            {
                return interfaceInfo->interfaceType;
            }
        }        

        return mlir::Type();
    }

    mlir::Type resolveTypeByName(mlir::Location location, StringRef name, const GenContext &genContext)
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

        if (!isEmbededType(name))
            emitError(location, "can't find type by name: ") << name;

        return mlir::Type();    
    }

    mlir::Value resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
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
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), getFullNamespaceName(name)), 
                    enumTypeInfo.first, 
                    enumTypeInfo.second), 
                enumTypeInfo.second);
        }

        if (getNamespaceMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            assert(namespaceInfo);
            auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
            return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto namespaceInfo = getNamespaceByFullName(fullName);
            if (namespaceInfo)
            {
                assert(namespaceInfo);
                auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
                return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
            }
        }

        auto type = resolveTypeByNameInNamespace(location, name, genContext);
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

    mlir::Value resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess,
                                          const GenContext &genContext)
    {
        if (fullNameGlobalsMap.count(name))
        {
            auto value = fullNameGlobalsMap.lookup(name);
            return globalVariableAccess(location, value, asAddess, genContext);
        }

        return mlir::Value();
    }

    mlir::Value globalVariableAccess(mlir::Location location, VariableDeclarationDOM::TypePtr value, bool asAddess,
                                     const GenContext &genContext)
    {
        if (!value->getType())
        {
            return mlir::Value();
        }

        auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(value->getType()),
                                                            value->getName(), ::mlir::IntegerAttr());
        if (asAddess)
        {
            return address;
        }

        return builder.create<mlir_ts::LoadOp>(location, value->getType(), address);
    }

    mlir::Value resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
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
            if (!isa<mlir_ts::ClassType>(genContext.thisType) && !isa<mlir_ts::ClassStorageType>(genContext.thisType))
            {
                return mlir::Value();
            }

            auto result = mlirGen(location, THIS_NAME, genContext);
            auto thisValue = V(result);

            auto classInfo =
                getClassInfoByFullName(mlir::cast<mlir_ts::ClassType>(genContext.thisType).getName().getValue());
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

    ValueOrLogicalResult mlirGen(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifier(location, name, genContext);
        if (value)
        {
            return value;
        }

        if (MLIRCustomMethods::isInternalFunctionName(compileOptions, name))
        {
            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, builder.getNoneType(), mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            symbOp->setAttr(BUILTIN_FUNC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return V(symbOp);
        }

        if (MLIRCustomMethods::isInternalObjectName(name))
        {
            mlir::Type type;

            if (name == "Symbol")
            {
                type = getSymbolType();
            }
            else
            {
                type = builder.getNoneType();
            }

            // set correct type
            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, type, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));            
            return V(symbOp);
        }

        // TODO: error, when we use  function_name(index: index) and index value is not provided in call function_name(index), index will be mistakenly tearted
        // as embeded type "index"
        if (!isEmbededType(name))
            emitError(location, "can't resolve name: ") << name;

        return mlir::failure();
    }

    TypeParameterDOM::TypePtr processTypeParameter(TypeParameterDeclaration typeParameter, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(typeParameter->name, stringAllocator);
        if (!namePtr.empty())
        {
            auto typeParameterDOM = std::make_shared<TypeParameterDOM>(namePtr.str());
            if (typeParameter->constraint)
            {
                typeParameterDOM->setConstraint(typeParameter->constraint);
            }

            if (typeParameter->_default)
            {
                typeParameterDOM->setDefault(typeParameter->_default);
            }

            return typeParameterDOM;
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::LogicalResult processTypeParameters(NodeArray<TypeParameterDeclaration> typeParameters,
                                              llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                              const GenContext &genContext)
    {
        for (auto typeParameter : typeParameters)
        {
            typeParams.push_back(processTypeParameter(typeParameter, genContext));
        }

        return mlir::success();
    }

    mlir::LogicalResult processTypeParametersFromFunctionParameters(SignatureDeclarationBase signatureDeclarationBase,
                                              llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                              const GenContext &genContext)
    {
        auto formalParams = signatureDeclarationBase->parameters;
        for (auto [index, arg] : enumerate(formalParams))
        {
            auto isBindingPattern = arg->name == SyntaxKind::ObjectBindingPattern || arg->name == SyntaxKind::ArrayBindingPattern;

            mlir::Type type;
            auto isMultiArgs = !!arg->dotDotDotToken;
            auto isOptional = !!arg->questionToken;            
            auto typeParameter = arg->type;

            auto location = loc(typeParameter);

            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                continue;
            }

            if (mth.isNoneType(type) && genContext.receiverFuncType && mth.isAnyFunctionType(genContext.receiverFuncType))
            {
                type = mth.getParamFromFuncRef(genContext.receiverFuncType, index);
                if (!type) continue;
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
            }

            if (mth.isNoneType(type))
            {
                if (!typeParameter && !initializer)
                {
                    auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
                    if (namePtr.empty())
                    {
                        namePtr = getArgumentName(index);
                    }                    

                    auto typeParamNamePtr = getParameterGenericTypeName(namePtr.str());      
                    auto &typeParameters = signatureDeclarationBase->typeParameters;
                    auto found = std::find_if(typeParameters.begin(), typeParameters.end(),
                                            [&](auto &paramItem) { return MLIRHelper::getName( paramItem->name) == typeParamNamePtr; });
                    if (found == typeParameters.end())
                    {
                        NodeFactory nf(NodeFactoryFlags::None);
                        auto wname = stows(typeParamNamePtr.str());
                        auto typeParameterDeclaration = nf.createTypeParameterDeclaration(undefined, nf.createIdentifier(wname), undefined, undefined);
                        signatureDeclarationBase->typeParameters.push_back(typeParameterDeclaration);

                        TypeNode typeNode = nf.createTypeReferenceNode(nf.createIdentifier(wname));
                        if (isMultiArgs)
                        {
                            typeNode = nf.createArrayTypeNode(typeNode);
                        }

                        if (isOptional)
                        {
                            typeNode = nf.createOptionalTypeNode(typeNode);
                        }

                        arg->type = typeNode;
                    } 

                    typeParams.push_back(std::make_shared<TypeParameterDOM>(typeParamNamePtr.str()));
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult processTypeArgumentsFromFunctionParameters(SignatureDeclarationBase signatureDeclarationBase,
                                              const GenContext &genContext)
    {
        auto isGenericTypes = false;
        auto formalParams = signatureDeclarationBase->parameters;
        for (auto [index, arg] : enumerate(formalParams))
        {
            auto isBindingPattern = arg->name == SyntaxKind::ObjectBindingPattern 
                || arg->name == SyntaxKind::ArrayBindingPattern;

            mlir::Type type;
            //auto isMultiArgs = !!arg->dotDotDotToken;
            //auto isOptional = !!arg->questionToken;            
            auto typeParameter = arg->type;

            auto location = loc(typeParameter);

            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                continue;
            }

            if (mth.isNoneType(type) && genContext.receiverFuncType && mth.isAnyFunctionType(genContext.receiverFuncType))
            {
                type = mth.getParamFromFuncRef(genContext.receiverFuncType, index);
                if (!type) continue;
                isGenericTypes |= mth.isGenericType(type);

                auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
                if (namePtr.empty())
                {
                    namePtr = getArgumentName(index);
                }                    

                auto typeParamNamePtr = getParameterGenericTypeName(namePtr.str());      
                auto typeParam = std::make_shared<TypeParameterDOM>(typeParamNamePtr.str());
                auto result = zipTypeParameterWithArgument(
                    location, const_cast<GenContext &>(genContext).typeParamsWithArgs, typeParam, type, false, genContext);
                EXIT_IF_FAILED(std::get<0>(result));
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
            }
        }

        return mlir::success();
    }    

    mlir::LogicalResult mlirGen(TypeAliasDeclaration typeAliasDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(typeAliasDeclarationAST->name, stringAllocator);
        if (!namePtr.empty())
        {
            auto hasExportModifier = getExportModifier(typeAliasDeclarationAST);

            if (typeAliasDeclarationAST->typeParameters.size() > 0)
            {
                llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
                if (mlir::failed(
                        processTypeParameters(typeAliasDeclarationAST->typeParameters, typeParameters, genContext)))
                {
                    return mlir::failure();
                }

                getGenericTypeAliasMap().insert({namePtr, {typeParameters, typeAliasDeclarationAST->type}});
            }
            else
            {
                GenContext typeAliasGenContext(genContext);
                auto type = getType(typeAliasDeclarationAST->type, typeAliasGenContext);
                if (!type)
                {
                    return mlir::failure();
                }

                getTypeAliasMap().insert({namePtr, type});

                if (hasExportModifier)
                {
                    addTypeDeclarationToExport(namePtr, currentNamespace, type);
                }
            }

            return mlir::success();
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    ValueOrLogicalResult mlirGenModuleReference(Node moduleReference, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)moduleReference;
        if (kind == SyntaxKind::QualifiedName)
        {
            return mlirGen(moduleReference.as<QualifiedName>(), genContext);
        }
        else if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(moduleReference.as<Identifier>(), genContext);
        }

        llvm_unreachable("not implemented");
    }

    mlir::LogicalResult mlirGen(ImportEqualsDeclaration importEqualsDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(importEqualsDeclarationAST->name);
        if (!name.empty())
        {
            auto result = mlirGenModuleReference(importEqualsDeclarationAST->moduleReference, genContext);
            auto value = V(result);
            if (auto namespaceOp = value.getDefiningOp<mlir_ts::NamespaceRefOp>())
            {
                getImportEqualsMap().insert({name, namespaceOp.getIdentifier()});
                return mlir::success();
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(value.getType()))
            {
                getImportEqualsMap().insert({name, classType.getName().getValue()});
                return mlir::success();
            }
            else if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(value.getType()))
            {
                getImportEqualsMap().insert({name, interfaceType.getName().getValue()});
                return mlir::success();
            }

            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(enumDeclarationAST->name, stringAllocator);
        if (namePtr.empty())
        {
            return mlir::failure();
        }

        SymbolTableScopeT varScope(symbolTable);

        SmallVector<mlir::Type> enumLiteralTypes;
        StringMap<mlir::Attribute> enumValues;

        auto appending = false;
        if (getEnumsMap().contains(namePtr))
        {
            auto dict = getEnumsMap().lookup(namePtr).second;
            for (auto key : dict)
            {
                enumValues[key.getName()] = key.getValue();
            }

            appending = true;
        }
        else
        {
            getEnumsMap().insert(
                { namePtr, { getEnumType().getElementType(), mlir::DictionaryAttr::get(builder.getContext(), {}) } });
        }

        auto &enumInfo = getEnumsMap()[namePtr];

        auto activeBits = 32;
        mlir::IntegerType::SignednessSemantics currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signless;
        llvm::APInt currentEnumValue(32, 0);
        for (auto enumMember : enumDeclarationAST->members)
        {
            auto location = loc(enumMember);

            auto memberNamePtr = MLIRHelper::getName(enumMember->name, stringAllocator);
            if (memberNamePtr.empty())
            {
                return mlir::failure();
            }

            mlir::Attribute enumValueAttr;
            if (enumMember->initializer)
            {
                GenContext enumValueGenContext(genContext);
                enumValueGenContext.allowConstEval = true;
                auto result = mlirGen(enumMember->initializer, enumValueGenContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto enumValue = V(result);

                LLVM_DEBUG(llvm::dbgs() << "\n!! enum member: [ " << memberNamePtr << " ] = [ " << enumValue << " ]\n");

                if (auto constOp = dyn_cast<mlir_ts::ConstantOp>(enumValue.getDefiningOp()))
                {
                    enumValueAttr = constOp.getValueAttr();
                    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(enumValueAttr))
                    {
                        if (intAttr.getType().isSignlessInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signless;
                        }
                        else if (intAttr.getType().isSignedInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signed;
                        }
                        else if (intAttr.getType().isUnsignedInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Unsigned;
                        }

                        currentEnumValue = intAttr.getValue();
                        auto currentActiveBits = (int)intAttr.getValue().getActiveBits();
                        if (currentActiveBits > activeBits)
                        {
                            activeBits = currentActiveBits;
                        }
                    }
                }
                else
                {
                    emitError(loc(enumMember->initializer))
                        << "enum member '" << memberNamePtr << "' must be constant";
                    return mlir::failure();
                }

                enumLiteralTypes.push_back(enumValue.getType());
                
                auto varDecl = std::make_shared<VariableDeclarationDOM>(memberNamePtr, enumValue.getType(), location);
                DECLARE(varDecl, enumValue);

            }
            else
            {
                if (appending && currentEnumValue == 0 && stage == Stages::Discovering && !enumValues.contains(memberNamePtr))
                {
                    emitError(loc(enumMember))
                        << "In an enum with multiple declarations, only one declaration can omit an initializer for its first enum element";                    
                    return mlir::failure();
                }

                auto typeInt = mlir::IntegerType::get(builder.getContext(), activeBits, currentEnumValueSigedness);
                enumValueAttr = builder.getIntegerAttr(typeInt, currentEnumValue);
                auto indexType = mlir_ts::LiteralType::get(enumValueAttr, typeInt);
                enumLiteralTypes.push_back(indexType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! enum member: " << memberNamePtr << " <- " << indexType << "\n");

                auto varDecl = std::make_shared<VariableDeclarationDOM>(memberNamePtr, indexType, location);
                auto enumVal = builder.create<mlir_ts::ConstantOp>(location, indexType, enumValueAttr);
                DECLARE(varDecl, enumVal);
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << namePtr << " value attr: " << enumValueAttr << "\n");

            enumValues[memberNamePtr] = enumValueAttr;

            // update enum to support req. access
            SmallVector<mlir::NamedAttribute> namedEnumValues;
            for (auto &key : enumValues)
            {
                namedEnumValues.push_back({builder.getStringAttr(key.first()), key.second});
            }

            enumInfo.second = mlir::DictionaryAttr::get(builder.getContext(), namedEnumValues /*adjustedEnumValues*/);

            currentEnumValue++;
        }

        auto location = loc(enumDeclarationAST);
        auto storeType = mth.getUnionTypeWithMerge(location, enumLiteralTypes);

        LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << namePtr << " storage type: " << storeType << "\n");

        // update enum to support req. access
        enumInfo.first = storeType;

        // register fullName for enum
        auto fullNamePtr = getFullNamespaceName(namePtr); 

        auto enumType = getEnumType(
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), fullNamePtr), 
                    enumInfo.first, 
                    enumInfo.second);

        EnumInfo::TypePtr newEnumPtr;
        if (fullNameEnumsMap.count(fullNamePtr))
        {
            newEnumPtr = fullNameEnumsMap.lookup(fullNamePtr);
            newEnumPtr->enumType = enumType;      
        }
        else
        {
            // register class
            newEnumPtr = std::make_shared<EnumInfo>();
            newEnumPtr->name = namePtr;
            newEnumPtr->fullName = fullNamePtr;
            newEnumPtr->elementNamespace = currentNamespace;      
            newEnumPtr->enumType = enumType;      
            fullNameEnumsMap.insert(fullNamePtr, newEnumPtr);        
        }

        if (getExportModifier(enumDeclarationAST))
        {
            addEnumDeclarationToExport(namePtr, currentNamespace, enumType);
        }

        return mlir::success();
    }

    mlir::LogicalResult registerGenericClass(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = className(classDeclarationAST, genContext);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericClassesMap.count(fullNamePtr))
            {
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

            mlirGenClassType(newGenericClassPtr, genContext);

            getGenericClassesMap().insert({namePtr, newGenericClassPtr});
            fullNameGenericClassesMap.insert(fullNamePtr, newGenericClassPtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(ClassDeclaration classDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(theModule.getBody());

        auto value = mlirGen(classDeclarationAST.as<ClassLikeDeclaration>(), genContext);
        return std::get<0>(value);
    }

    ValueOrLogicalResult mlirGen(ClassExpression classExpressionAST, const GenContext &genContext)
    {
        std::string fullName;

        // go to root
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            auto [result, fullNameRet] = mlirGen(classExpressionAST.as<ClassLikeDeclaration>(), genContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            fullName = fullNameRet;
        }

        auto location = loc(classExpressionAST);

        auto classInfo = getClassInfoByFullName(fullName);
        if (classInfo)
        {
            if (classInfo->isDeclaration)
            {
                auto undefClass = builder.create<mlir_ts::UndefOp>(location, classInfo->classType);
                return V(undefClass);
            }
            else
            {
                auto classValue = builder.create<mlir_ts::ClassRefOp>(
                    location, classInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));

                // TODO: find out if you need to pass generics info, typeParams + typeArgs
                return NewClassInstance(location, classValue, undefined, undefined, false, genContext);
            }
        }

        return mlir::failure();
    }

    bool testProcessingState(ClassInfo::TypePtr &newClassPtr, ProcessingStages state, const GenContext &genContext) {
        return (genContext.allowPartialResolve)
            ? newClassPtr->processingAtEvaluation >= state
            : newClassPtr->processing >= state;
    }

    void setProcessingState(ClassInfo::TypePtr &newClassPtr, ProcessingStages state, const GenContext &genContext) {
        if (genContext.allowPartialResolve)
        {
            newClassPtr->processingAtEvaluation = state;
        }
        else
        {
            newClassPtr->processing = state;
        }        
    }

    std::pair<mlir::LogicalResult, mlir::StringRef> mlirGen(ClassLikeDeclaration classDeclarationAST,
                                                            const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        auto isGenericClass = classDeclarationAST->typeParameters.size() > 0;
        if (isGenericClass && genContext.typeParamsWithArgs.size() == 0)
        {
            return {registerGenericClass(classDeclarationAST, genContext), ""};
        }

        auto newClassPtr = mlirGenClassInfo(classDeclarationAST, genContext);
        if (!newClassPtr)
        {
            return {mlir::failure(), ""};
        }

        // do not process specialized class second time;
        if (isGenericClass && genContext.typeParamsWithArgs.size() > 0)
        {
            // TODO: investigate why classType is provided already for class
            if (testProcessingState(newClassPtr, ProcessingStages::Processing, genContext))
            {
                return {mlir::success(), newClassPtr->classType.getName().getValue()};
            }
        }

        setProcessingState(newClassPtr, ProcessingStages::Processing, genContext);

        auto location = loc(classDeclarationAST);

        if (mlir::succeeded(mlirGenClassType(newClassPtr, genContext)))
        {
            newClassPtr->typeParamsWithArgs = genContext.typeParamsWithArgs;
        }

        // if this is generic specialized class then do not generate code for it
        if (mth.isGenericType(newClassPtr->classType))
        {
            return {mlir::success(), newClassPtr->classType.getName().getValue()};
        }

        // init this type (needed to use in property evaluations)
        GenContext classGenContext(genContext);
        classGenContext.thisType = newClassPtr->classType;
        classGenContext.specialization = isGenericClass;

        // we need THIS in params
        SymbolTableScopeT varScope(symbolTable);

        setProcessingState(newClassPtr, ProcessingStages::ProcessingStorageClass, genContext);
        if (mlir::failed(mlirGenClassStorageType(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInStorageClass, genContext);
            return {mlir::failure(), ""};
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessedStorageClass, genContext);

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
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        if (!newClassPtr->isStatic)
        {
            mlirGenClassDefaultConstructor(classDeclarationAST, newClassPtr, classGenContext);
        }

#ifdef ENABLE_RTTI
        if (!newClassPtr->isStatic)
        {
            // INFO: .instanceOf must be first element in VTable for Cast Any
            mlirGenClassInstanceOfMethod(classDeclarationAST, newClassPtr, classGenContext);
        }
#endif

#if ENABLE_TYPED_GC
        auto enabledGC = !compileOptions.disableGC;
        if (enabledGC && !newClassPtr->isStatic)
        {
            mlirGenClassTypeBitmap(location, newClassPtr, classGenContext);
            mlirGenClassTypeDescriptorField(location, newClassPtr, classGenContext);
        }
#endif

        if (!newClassPtr->isStatic)
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
            return {mlir::failure(), ""};
        }

        // generate vtable for interfaces in base class
        if (mlir::failed(mlirGenClassBaseInterfaces(location, newClassPtr, classGenContext)))
        {
            return {mlir::failure(), ""};
        }

        // generate vtable for interfaces
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClauseImplements(classDeclarationAST, newClassPtr, heritageClause,
                                                                  classGenContext)))
            {
                return {mlir::failure(), ""};
            }
        }

        if (!newClassPtr->isStatic)
        {
            if (mlir::failed(mlirGenClassVirtualTableDefinition(location, newClassPtr, classGenContext)))
            {
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

    void appendSpecializedTypeNames(std::string &name, llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                    const GenContext &genContext)
    {
        name.append("<");
        auto next = false;
        for (auto typeParam : typeParams)
        {
            if (next)
            {
                name.append(",");
            }

            auto type = getResolveTypeParameter(typeParam->getName(), false, genContext);
            if (type)
            {
                llvm::raw_string_ostream s(name);
                s << type;
            }
            else
            {
                name.append(typeParam->getName());
            }

            next = true;
        }

        name.append(">");
    }

    std::string getSpecializedClassName(GenericClassInfo::TypePtr geneticClassPtr, const GenContext &genContext)
    {
        auto name = geneticClassPtr->fullName.str();
        if (genContext.typeParamsWithArgs.size())
        {
            appendSpecializedTypeNames(name, geneticClassPtr->typeParams, genContext);
        }

        return name;
    }

    mlir_ts::ClassType getSpecializationClassType(GenericClassInfo::TypePtr genericClassPtr,
                                                  const GenContext &genContext)
    {
        auto fullSpecializedClassName = getSpecializedClassName(genericClassPtr, genContext);
        auto classInfoType = getClassInfoByFullName(fullSpecializedClassName);
        classInfoType->originClassType = genericClassPtr->classType;
        assert(classInfoType);
        return classInfoType->classType;
    }

    std::string className(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = getNameWithArguments(classDeclarationAST, genContext);
        if (classDeclarationAST == SyntaxKind::ClassExpression)
        {
            NodeFactory nf(NodeFactoryFlags::None);
            classDeclarationAST->name = nf.createIdentifier(stows(name));
        }

        return name;
    }

    ClassInfo::TypePtr mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        return mlirGenClassInfo(className(classDeclarationAST, genContext), classDeclarationAST, genContext);
    }

    ClassInfo::TypePtr mlirGenClassInfo(const std::string &name, ClassLikeDeclaration classDeclarationAST,
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
            MLIRHelper::iterateDecorators(classDeclarationAST, [&](std::string name, SmallVector<std::string> args) {
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
            });

            getClassesMap().insert({namePtr, newClassPtr});
            fullNameClassesMap.insert(fullNamePtr, newClassPtr);
        }

        return newClassPtr;
    }

    template <typename T> mlir::LogicalResult mlirGenClassType(T newClassPtr, const GenContext &genContext)
    {
        if (newClassPtr)
        {
            auto classFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), newClassPtr->fullName);
            newClassPtr->classType = getClassType(classFullNameSymbol, getClassStorageType(classFullNameSymbol));
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenClassCheckIfDeclaration(mlir::Location location,
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

    mlir::LogicalResult mlirGenClassTypeSetFields(ClassInfo::TypePtr newClassPtr,
                                                  SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        if (newClassPtr)
        {
            mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType()).setFields(fieldInfos);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenClassStorageType(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
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

#if ENABLE_RTTI
        if (newClassPtr->isDynamicImport)
        {
            mlirGenCustomRTTIDynamicImport(location, classDeclarationAST, newClassPtr, genContext);
        }
        else if (!newClassPtr->isStatic)
        {
            newClassPtr->hasVirtualTable = true;
            mlirGenCustomRTTI(location, classDeclarationAST, newClassPtr, genContext);
        }
#endif

        if (!newClassPtr->isStatic)
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

        if (newClassPtr->getHasVirtualTableVariable())
        {
            auto fieldId = MLIRHelper::TupleFieldName(VTABLE_NAME, builder.getContext());
            if (fieldInfos.size() == 0 || fieldInfos.front().id != fieldId)
            {
                fieldInfos.insert(fieldInfos.begin(), {fieldId, getOpaqueType()});
            }
        }

        mlirGenClassTypeSetFields(newClassPtr, fieldInfos);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassStaticFields(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
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

    mlir::LogicalResult mlirGenClassMembers(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
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

    void clearMembersProcessStates(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr) {
        // to be able to run next time, code succeeded, and we know where to continue from
        for (auto &classMember : newClassPtr->extraMembers)
        {
            classMember->processed = false;
        }

        for (auto &classMember : classDeclarationAST->members)
        {
            classMember->processed = false;
        }

        for (auto &classMember : newClassPtr->extraMembersPost)
        {
            classMember->processed = false;
        }
    }

    void saveMembersProcessStates(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr, 
            llvm::SmallVector<bool> &membersProcessStates) {
        // we need only members from class AST (not extraMembers and not extraMembersPost)
        for (auto &classMember : classDeclarationAST->members)
        {
            membersProcessStates.push_back(classMember->processed);
        }
    }

    void restoreMembersProcessStates(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr, 
            llvm::SmallVector<bool> &membersProcessStates) {
        for (auto [index, classMember] : enumerate(classDeclarationAST->members))
        {
            classMember->processed = membersProcessStates[index];
        }

        membersProcessStates.clear();
    }    

    mlir::LogicalResult mlirGenClassHeritageClause(ClassLikeDeclaration classDeclarationAST,
                                                   ClassInfo::TypePtr newClassPtr, HeritageClause heritageClause,
                                                   SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);

        if (heritageClause->token == SyntaxKind::ExtendsKeyword)
        {
            auto &baseClassInfos = newClassPtr->baseClasses;

            for (auto &extendingType : heritageClause->types)
            {
                auto result = mlirGen(extendingType, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto baseType = V(result);
                mlir::TypeSwitch<mlir::Type>(baseType.getType())
                    .template Case<mlir_ts::ClassType>([&](auto baseClassType) {
                        auto baseName = baseClassType.getName().getValue();
                        auto fieldId = MLIRHelper::TupleFieldName(baseName, builder.getContext());
                        fieldInfos.push_back({fieldId, baseClassType.getStorageType()});

                        auto classInfo = getClassInfoByFullName(baseName);
                        if (std::find(baseClassInfos.begin(), baseClassInfos.end(), classInfo) == baseClassInfos.end())
                        {
                            baseClassInfos.push_back(classInfo);
                        }
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
            return mlir::success();
        }

        if (heritageClause->token == SyntaxKind::ImplementsKeyword)
        {
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
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
        }

        return mlir::success();
    }

    Node getFieldNameForAccessor(Node name) {
        auto nameStr = MLIRHelper::getName(name);
        nameStr.insert(0, "#__");

        NodeFactory nf(NodeFactoryFlags::None);
        auto newName = nf.createIdentifier(stows(nameStr.c_str()));
        return newName;
    }

    mlir::LogicalResult mlirGenClassDataFieldAccessor(mlir::Location location, ClassInfo::TypePtr newClassPtr, 
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

                    getTypeAliasMap().insert({fieldTypeAlias, typeIfNotProvided});
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

    mlir::LogicalResult mlirGenClassDataFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                    PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto name = propertyDeclaration->name;
        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            name = getFieldNameForAccessor(name);
        }
        
        auto fieldId = TupleFieldName(name, genContext);

        auto [type, init, typeProvided] = evaluateTypeAndInit(propertyDeclaration, genContext);
        if (init)
        {
            newClassPtr->hasInitializers = true;
            type = mth.wideStorageType(type);
        }

        LLVM_DEBUG(dbgs() << "\n!! class field: " << fieldId << " type: " << type << "");

        auto hasType = !!propertyDeclaration->type;
        if (mth.isNoneType(type))
        {
            if (hasType)
            {
                return mlir::failure();
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

        fieldInfos.push_back({fieldId, type});

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, type, genContext);
            EXIT_IF_FAILED(res)
        }        

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassStaticFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto isPublic = hasModifier(propertyDeclaration, SyntaxKind::PublicKeyword);
        auto name = propertyDeclaration->name;

        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            isPublic = false;
            name = getFieldNameForAccessor(name);
        }

        auto fieldId = TupleFieldName(name, genContext);

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

        auto &staticFieldInfos = newClassPtr->staticFields;
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1);

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, staticFieldType, genContext);
            EXIT_IF_FAILED(res)
        }  

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassStaticFieldMemberDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto fieldId = TupleFieldName(propertyDeclaration->name, genContext);

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
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1);

        return mlir::success();
    }    

    mlir::LogicalResult mlirGenClassConstructorPublicDataFieldMembers(mlir::Location location, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
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

            auto [type, init, typeProvided] = getTypeAndInit(parameter, genContext);

            LLVM_DEBUG(dbgs() << "\n+++ class auto-gen field: " << fieldId << " type: " << type << "");
            if (mth.isNoneType(type))
            {
                return mlir::failure();
            }

            fieldInfos.push_back({fieldId, type});
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassProcessClassPropertyByFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember)
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
            }
        }

        if (newClassPtr->isStatic)
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

    mlir::LogicalResult mlirGenClassFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
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
            if (mlir::failed(mlirGenClassConstructorPublicDataFieldMembers(location, fieldInfos, constructorDeclaration, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenForwardDeclaration(const std::string &funcName, mlir_ts::FunctionType funcType,
                                                  bool isStatic, bool isVirtual, bool isAbstract,
                                                  ClassInfo::TypePtr newClassPtr, int orderWeight, const GenContext &genContext)
    {
        if (newClassPtr->getMethodIndex(funcName) < 0)
        {
            return mlir::success();
        }

        mlir_ts::FuncOp dummyFuncOp;
        newClassPtr->methods.push_back({funcName, funcType, dummyFuncOp, isStatic,
                                        isVirtual || isAbstract, isAbstract, -1, orderWeight});
        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassNew(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
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

        //LLVM_DEBUG(printDebug(generatedNew););

        newClassPtr->extraMembers.push_back(generatedNew);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasInitializers && !newClassPtr->hasConstructor)
        {
            // create constructor
            newClassPtr->hasConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;

            if (!newClassPtr->baseClasses.empty())
            {
                auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                statements.push_back(nf.createExpressionStatement(callSuper));
            }

            auto body = nf.createBlock(statements, /*multiLine*/ false);

            auto generatedConstructor = nf.createConstructorDeclaration(undefined, undefined, body);
            newClassPtr->extraMembers.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassDefaultStaticConstructor(ClassLikeDeclaration classDeclarationAST,
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

    // to support crearting classes in Stack
    mlir::LogicalResult mlirGenClassSizeStaticField(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
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
#ifdef WIN32                
                varClass.comdat = Select::ExactMatch;
#else
                varClass.comdat = Select::Any;
#endif                
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

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1);

        return mlir::success();    
    }

    void pushStaticField(llvm::SmallVector<StaticFieldInfo> &staticFieldInfos, mlir::Attribute fieldId, mlir::Type staticFieldType, StringRef fullClassStaticFieldName, int index)
    {
        if (!llvm::any_of(staticFieldInfos, [&](auto& field) 
            { 
                auto foundField = field.id == fieldId;
                if (foundField)
                {
                    // update field type if different
                    if (field.type != staticFieldType)
                    {
                        assert(false);
                        field.type = staticFieldType;
                    }
                }
                
                return foundField; 
            }))
        {
            staticFieldInfos.push_back({fieldId, staticFieldType, fullClassStaticFieldName, index});
        }        
    }

    // INFO: you can't use standart Static Field declarastion because of RTTI should be declared before used
    // example: C:/dev/TypeScriptCompiler/tsc/test/tester/tests/dependencies.ts
    mlir::LogicalResult mlirGenCustomRTTI(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        auto &staticFieldInfos = newClassPtr->staticFields;

        auto fieldId = MLIRHelper::TupleFieldName(RTTI_NAME, builder.getContext());

        // register global
        auto fullClassStaticFieldName = concat(newClassPtr->fullName, RTTI_NAME);

        auto staticFieldType = getStringType();

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            // prevent double generating
            VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
            varClass.isExport = newClassPtr->isExport;
            varClass.isImport = newClassPtr->isImport;
            varClass.isPublic = newClassPtr->isPublic;
            registerVariable(
                location, fullClassStaticFieldName, true, varClass,
                [&](mlir::Location location, const GenContext &genContext) {
                    if (newClassPtr->isDeclaration)
                    {
                        return std::make_tuple(staticFieldType, mlir::Value(), TypeProvided::Yes);
                    }

                    mlir::Value init = builder.create<mlir_ts::ConstantOp>(location, staticFieldType,
                                                                            getStringAttr(newClassPtr->fullName.str()));
                    return std::make_tuple(staticFieldType, init, TypeProvided::Yes);
                },
                genContext);
        }

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1);

        return mlir::success();
    }

    // INFO: you can't use standart Static Field declarastion because of RTTI should be declared before used
    // example: C:/dev/TypeScriptCompiler/tsc/test/tester/tests/dependencies.ts
    mlir::LogicalResult mlirGenCustomRTTIDynamicImport(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        return mlirGenStaticFieldDeclarationDynamicImport(location, newClassPtr, RTTI_NAME, getStringType(), genContext);
    }

#ifdef ENABLE_TYPED_GC
    StringRef getTypeBitmapMethodName(ClassInfo::TypePtr newClassPtr)
    {
        return concat(newClassPtr->fullName, TYPE_BITMAP_NAME);
    }

    StringRef getTypeDescriptorFieldName(ClassInfo::TypePtr newClassPtr)
    {
        return concat(newClassPtr->fullName, TYPE_DESCR_NAME);
    }

    mlir::LogicalResult mlirGenClassTypeDescriptorField(mlir::Location location, ClassInfo::TypePtr newClassPtr,
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
                    auto init =
                        builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(), mth.getI64AttrValue(0));
                    return std::make_tuple(init.getType(), init, TypeProvided::Yes);
                },
                genContext);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassTypeBitmap(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                               const GenContext &genContext)
    {
        // no need to generate
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        MLIRCodeLogic mcl(builder);

        // register global
        auto name = TYPE_BITMAP_NAME;
        auto fullClassStaticFieldName = getTypeBitmapMethodName(newClassPtr);

        auto funcType = getFunctionType({}, builder.getI64Type(), false);

        mlirGenFunctionBody(
            location, name, fullClassStaticFieldName, funcType,
            [&](mlir::Location location, const GenContext &genContext) {
                auto bitmapValueType = mth.getTypeBitmapValueType();

                auto nullOp = builder.create<mlir_ts::NullOp>(location, getNullType());
                CAST_A(classNull, location, newClassPtr->classType, nullOp, genContext);

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

                    CAST_A(calcIndex32, location, mth.getStructIndexType(), calcIndex, genContext);

                    auto elemRef = builder.create<mlir_ts::PointerOffsetRefOp>(
                        location, mlir_ts::RefType::get(bitmapValueType), arrayValue, calcIndex32);

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

                auto typeDescr = builder.create<mlir_ts::GCMakeDescriptorOp>(location, builder.getI64Type(), arrayValue,
                                                                             sizeOfTypeInBitmapTypes);

                auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
                builder.create<mlir_ts::ReturnValOp>(location, typeDescr, retVarInfo.first);
                return ValueOrLogicalResult(mlir::success());
            },
            genContext);

        return mlir::success();
    }

#endif

    mlir::LogicalResult mlirGenClassInstanceOfMethod(ClassLikeDeclaration classDeclarationAST,
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

    ValueOrLogicalResult mlirGenCreateInterfaceVTableForClass(mlir::Location location, ClassInfo::TypePtr newClassPtr,
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

    ValueOrLogicalResult mlirGenCreateInterfaceVTableForObject(mlir::Location location, mlir::Value in, 
            mlir_ts::ObjectType objectType, InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        auto fullObjectInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(
                mlirGenObjectVirtualTableDefinitionForInterface(location, objectType, newInterfacePtr, genContext)))
        {
            auto globalVTableRefValue = resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);

            // we need to update methods references in VTable with functions from object;
            if (newInterfacePtr->methods.size() > 0) {

                mlir_ts::TupleType storeType;
                if (auto objectStoreType = dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
                {
                    storeType = mlir_ts::TupleType::get(builder.getContext(), objectStoreType.getFields());
                }
                else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(objectType.getStorageType()))
                {
                    storeType = tupleType;
                }
                else
                {
                    return mlir::failure();
                }

                // match VTable
                // 1) clone vtable
                auto vtableType = mlir::cast<mlir_ts::TupleType>(mlir::cast<mlir_ts::RefType>(globalVTableRefValue.getType()).getElementType());
                auto valueVTable = builder.create<mlir_ts::LoadOp>(location, vtableType, globalVTableRefValue);
                auto varVTable = builder.create<mlir_ts::VariableOp>(location, globalVTableRefValue.getType(), valueVTable, 
                    builder.getBoolAttr(false), builder.getIndexAttr(0));

                for (auto& method : newInterfacePtr->methods)
                {
                    auto index = mth.getFieldIndexByFieldName(storeType, builder.getStringAttr(method.name));
                    if (index == -1)
                    {
                        return mlir::failure();
                    }

                    auto fieldInfo = mth.getFieldInfoByIndex(storeType, index);

                    auto methodRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(fieldInfo.type), in, index);

                    LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable method: " << method.name
                                            << "\n\t object method ref: " << V(methodRef) << "\n\n";);

                    // where to save
                    auto fieldInfoVT = mth.getFieldInfoByIndex(vtableType, method.virtualIndex);
                    auto methodRefVT = builder.create<mlir_ts::PropertyRefOp>(location, fieldInfoVT.type, varVTable, method.virtualIndex);

                    LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable method: " << method.name
                                            << "\n\t vtable method ref: " << V(methodRefVT) << "\n\n";);                    
                    
                    builder.create<mlir_ts::LoadSaveOp>(location, methodRefVT, methodRef);   
                }       

                // patched VTable
                return V(varVTable);         
            }

            return globalVTableRefValue;
        }

        return mlir::failure();
    }

    StringRef interfaceVTableNameForClass(ClassInfo::TypePtr newClassPtr, InterfaceInfo::TypePtr newInterfacePtr)
    {
        return concat(newClassPtr->fullName, newInterfacePtr->fullName, VTABLE_NAME);
    }

    StringRef interfaceVTableNameForObject(mlir_ts::ObjectType objectType, InterfaceInfo::TypePtr newInterfacePtr)
    {
        std::stringstream ss;
        ss << hash_value(objectType);

        return concat(newInterfacePtr->fullName, ss.str().c_str(), VTABLE_NAME);
    }

    mlir::LogicalResult getInterfaceVirtualTableForObject(mlir::Location location, mlir_ts::TupleType tupleStorageType,
                                                          InterfaceInfo::TypePtr newInterfacePtr,
                                                          SmallVector<VirtualMethodOrFieldInfo> &virtualTable,
                                                          bool suppressErrors = false)
    {
        return mth.getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable, suppressErrors);
    }

    mlir::LogicalResult mlirGenObjectVirtualTableDefinitionForInterface(mlir::Location location,
                                                                        mlir_ts::ObjectType objectType,
                                                                        InterfaceInfo::TypePtr newInterfacePtr,
                                                                        const GenContext &genContext)
    {

        MLIRCodeLogic mcl(builder);

        auto storeType = objectType.getStorageType();

        // TODO: should object accept only ObjectStorageType?
        if (auto objectStoreType = dyn_cast<mlir_ts::ObjectStorageType>(storeType))
        {
            storeType = mlir_ts::TupleType::get(builder.getContext(), objectStoreType.getFields());
        }

        auto tupleStorageType = mlir::cast<mlir_ts::TupleType>(mth.convertConstTupleTypeToTupleType(storeType));

        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable);
        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) {
                // build vtable from names of methods

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        if (!methodOrField.isMissing)
                        {
                            // TODO: test cast result
                            auto objectNull = cast(location, objectType, nullObj, genContext);
                            auto fieldValue = mlirGenPropertyAccessExpression(location, objectNull,
                                                                              methodOrField.fieldInfo.id, genContext);
                            assert(fieldValue);
                            auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);

                            LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable field: " << methodOrField.fieldInfo.id
                                                    << "\n\t type: " << methodOrField.fieldInfo.type
                                                    << "\n\t provided data: " << fieldRef << "\n\n";);

                            if (isa<mlir_ts::BoundRefType>(fieldRef.getType()))
                            {
                                fieldRef = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), fieldRef,
                                                genContext);
                            }
                            else
                            {
                                assert(mlir::cast<mlir_ts::RefType>(fieldRef.getType()).getElementType() ==
                                       methodOrField.fieldInfo.type);
                            }

                            // insert &(null)->field
                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, fieldRef, vtableValue,
                                MLIRHelper::getStructIndex(builder, fieldIndex));
                        }
                        else
                        {
                            // null value, as missing field/method
                            // auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                            auto negative1 = builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(),
                                                                                 mth.getI64AttrValue(-1));
                            auto castedNull = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type),
                                                   negative1, genContext);
                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, castedNull, vtableValue,
                                MLIRHelper::getStructIndex(builder, fieldIndex));
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented yet");
                        /*
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(),
                        methodOrField.methodInfo.funcOp.getSymName()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      MLIRHelper::getStructIndex(rewriter, fieldIndex));
                        */
                    }

                    fieldIndex++;
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location,
                                                                       ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr,
                                                                       const GenContext &genContext)
    {

        MLIRCodeLogic mcl(builder);

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        // TODO: ...
        auto classStorageType = mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType());

        llvm::SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> std::pair<mlir_ts::FieldInfo, mlir::LogicalResult> {
                auto found = false;
                auto foundField = newClassPtr->findField(id, found);
                if (!found || fieldType != foundField.type)
                {
                    if (!found && !isConditional || found)
                    {
                        emitError(location)
                            << "field type not matching for '" << id << "' for interface '" << newInterfacePtr->fullName
                            << "' in class '" << newClassPtr->fullName << "'";

                        return {emptyFieldInfo, mlir::failure()};
                    }

                    return {emptyFieldInfo, mlir::success()};
                }

                return {foundField, mlir::success()};
            },
            [&](std::string name, mlir_ts::FunctionType funcType, bool isConditional, int interfacePosIndex) -> std::pair<MethodInfo &, mlir::LogicalResult> {
                auto foundMethodPtr = newClassPtr->findMethod(name);
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

                auto foundMethodFunctionType = mlir::cast<mlir_ts::FunctionType>(foundMethodPtr->funcOp.getFunctionType());

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

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        auto classNull = cast(location, newClassPtr->classType, nullObj, genContext);
                        auto fieldValue = mlirGenPropertyAccessExpression(location, classNull,
                                                                          methodOrField.fieldInfo.id, genContext);
                        auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);
                        if (!fieldRef)
                        {
                            emitError(location) << "can't find reference for field: " << methodOrField.fieldInfo.id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        // insert &(null)->field
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, fieldRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else
                    {
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcOp.getFunctionType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                         methodOrField.methodInfo.funcOp.getSymName()));

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, methodConstName, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }

                    fieldIndex++;
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return mlir::success();
    }

    MethodInfo *generateSynthMethodToCallNewCtor(mlir::Location location, ClassInfo::TypePtr newClassPtr, InterfaceInfo::TypePtr newInterfacePtr, 
                                            mlir_ts::FunctionType funcType, int interfacePosIndex, const GenContext &genContext)
    {
        auto fullClassStaticName = generateSynthMethodToCallNewCtor(location, newClassPtr, newInterfacePtr->fullName, interfacePosIndex, funcType, 1, genContext);
        return newClassPtr->findMethod(fullClassStaticName);
    }    

    std::string generateSynthMethodToCallNewCtor(mlir::Location location, ClassInfo::TypePtr newClassPtr, StringRef sourceOwnerName, int posIndex, 
                                            mlir_ts::FunctionType funcType, int skipFuncParams, const GenContext &genContext)
    {
        auto fullClassStaticName = concat(newClassPtr->fullName, sourceOwnerName, NEW_CTOR_METHOD_NAME, posIndex);

        auto retType = mth.getReturnTypeFromFuncRef(funcType);
        if (!retType)
        {
            return "";
        }

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            GenContext funcGenContext(genContext);
            funcGenContext.clearScopeVars();
            funcGenContext.thisType = newClassPtr->classType;
            funcGenContext.disableSpreadParams = true;

            auto result = mlirGenFunctionBody(
                location, NEW_CTOR_METHOD_NAME, fullClassStaticName, funcType,
                [&](mlir::Location location, const GenContext &genContext) {
                    NodeFactory nf(NodeFactoryFlags::None);

                    NodeArray<Expression> argumentsArray;
                    //auto skip = 1;
                    auto skip = skipFuncParams;
                    auto index = 0;
                    for (auto &paramType : funcType.getInputs())
                    {
                        (void)paramType;

                        if (skip-- > 0) 
                        {
                            continue;
                        }

                        std::string paramName("p");
                        paramName += std::to_string(index++);
                        argumentsArray.push_back(nf.createIdentifier(stows(paramName)));
                    }

                    auto newInst = nf.createNewExpression(nf.createToken(SyntaxKind::ThisKeyword), undefined, argumentsArray);
                    auto instRes = mlirGen(newInst, funcGenContext);
                    EXIT_IF_FAILED(instRes);
                    auto instVal = V(instRes);
                    auto castToRet = cast(location, retType, instVal, funcGenContext);
                    EXIT_IF_FAILED(castToRet);
                    auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
                    if (retVarInfo.second)
                    {
                        builder.create<mlir_ts::ReturnValOp>(location, castToRet, retVarInfo.first);
                    }
                    else
                    {
                        return mlir::failure();
                    }

                    return mlir::success();
                },
                funcGenContext, skipFuncParams/*to skip This*/);        

            if (mlir::failed(result))
            {
                return "";
            }
        }

        // register method in info
        if (newClassPtr->getMethodIndex(fullClassStaticName) < 0)
        {
            auto funcOp = mlir_ts::FuncOp::create(location, fullClassStaticName, funcType);

            auto &methodInfos = newClassPtr->methods;
            methodInfos.push_back(
                {fullClassStaticName.str(), funcType, funcOp, true, false, false, -1, posIndex});
        }        

        return fullClassStaticName.str();
    }

    mlir::LogicalResult mlirGenClassBaseInterfaces(mlir::Location location, ClassInfo::TypePtr newClassPtr,
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

    mlir::LogicalResult mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST,
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
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrFieldInfo> &virtualTable)
    {
        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isField)
            {
                fields.push_back({vtableRecord.fieldInfo.id, mlir_ts::RefType::get(vtableRecord.fieldInfo.type)});
            }
            else
            {
                fields.push_back({MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()),
                                  vtableRecord.methodInfo.funcOp ? vtableRecord.methodInfo.funcOp.getFunctionType()
                                                                 : vtableRecord.methodInfo.funcType});
            }
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &virtualTable)
    {
        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isInterfaceVTable)
            {
                fields.push_back({MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()), getOpaqueType()});
            }
            else
            {
                if (!vtableRecord.isStaticField)
                {
                    fields.push_back({MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()),
                                      vtableRecord.methodInfo.funcOp ? vtableRecord.methodInfo.funcOp.getFunctionType()
                                                                     : vtableRecord.methodInfo.funcType});
                }
                else
                {
                    fields.push_back(
                        {vtableRecord.staticFieldInfo.id, mlir_ts::RefType::get(vtableRecord.staticFieldInfo.type)});
                }
            }
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinition(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                           const GenContext &genContext)
    {
        if (!newClassPtr->getHasVirtualTable() || newClassPtr->isAbstract)
        {
            return mlir::success();
        }
       
        // TODO: ...
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
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
                MLIRCodeLogic mcl(builder);
                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto vtRecord : virtualTable)
                {
                    if (vtRecord.isInterfaceVTable)
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
                        mlir::Value methodOrFieldNameRef;
                        if (!vtRecord.isStaticField)
                        {
                            if (vtRecord.methodInfo.isAbstract)
                            {
                                emitError(location) << "Abstract method '" << vtRecord.methodInfo.name <<  "' is not implemented in '" << newClassPtr->name << "'";
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No}; 
                            }

                            methodOrFieldNameRef = builder.create<mlir_ts::SymbolRefOp>(
                                location, vtRecord.methodInfo.funcOp.getFunctionType(),
                                mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                             vtRecord.methodInfo.funcOp.getSymName()));
                        }
                        else
                        {
                            methodOrFieldNameRef = builder.create<mlir_ts::SymbolRefOp>(
                                location, mlir_ts::RefType::get(vtRecord.staticFieldInfo.type),
                                mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                             vtRecord.staticFieldInfo.globalVariableName));
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

    struct ClassMethodMemberInfo
    {
        ClassMethodMemberInfo(ClassInfo::TypePtr newClassPtr, ClassElement classMember) : newClassPtr(newClassPtr), classMember(classMember)
        {
            isConstructor = classMember == SyntaxKind::Constructor;
            isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
            isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
            //auto isPrivate = hasModifier(classMember, SyntaxKind::PrivateKeyword);
            //auto isProtected = hasModifier(classMember, SyntaxKind::ProtectedKeyword);
            auto isPublic = hasModifier(classMember, SyntaxKind::PublicKeyword);

            isExport = newClassPtr->isExport && (isConstructor || isPublic);
            isImport = newClassPtr->isImport && (isConstructor || isPublic);
            isForceVirtual = (classMember->internalFlags & InternalFlags::ForceVirtual) == InternalFlags::ForceVirtual;
    #ifdef ALL_METHODS_VIRTUAL
            isForceVirtual |= !isConstructor;
    #endif
            isVirtual = isForceVirtual;
        };

        bool isFunctionLike()
        {
            return classMember == SyntaxKind::MethodDeclaration || isConstructor || classMember == SyntaxKind::GetAccessor ||
                classMember == SyntaxKind::SetAccessor;
        }

        std::string getName()
        {
            return propertyName.empty() ? methodName : propertyName;
        }

        StringRef getFuncName()
        {
            return funcOp.getName();
        }

        mlir_ts::FunctionType getFuncType()
        {
            return funcOp.getFunctionType();
        }

        void setFuncOp(mlir_ts::FuncOp funcOp_)
        {
            funcOp = funcOp_;
        }

        bool registerClassMethodMember(mlir::Location location, int orderWeight)
        {
            auto &methodInfos = newClassPtr->methods;

            if (newClassPtr->getMethodIndex(methodName) < 0)
            {
                methodInfos.push_back(
                    {methodName, getFuncType(), funcOp, isStatic, isAbstract || isVirtual, isAbstract, -1, orderWeight});
            }

            if (propertyName.size() > 0)
            {
                addAccessor();
            }

            if (newClassPtr->indexes.size() > 0)
            {
                if (methodName == "get")
                {
                    auto &indexer = newClassPtr->indexes.front();
                    auto getFuncType = funcOp.getFunctionType();
                    auto signatureFromGetFunc = 
                        mlir_ts::FunctionType::get(
                            indexer.indexSignature.getContext(), 
                            getFuncType.getInputs().drop_front(), 
                            getFuncType.getResults(), 
                            false);                    
                    if (indexer.indexSignature != signatureFromGetFunc)
                    {
                        emitError(location) << "'get' method is not matching 'index' definition";
                        return false;
                    }

                    indexer.get = funcOp;
                }
                else if (methodName == "set")
                {
                    auto &indexer = newClassPtr->indexes.front();
                    auto setFuncType = funcOp.getFunctionType();
                    auto signatureFromSetFunc = 
                        mlir_ts::FunctionType::get(
                            indexer.indexSignature.getContext(), 
                            setFuncType.getInputs().drop_front().drop_back(), 
                            { setFuncType.getInputs().back() }, 
                            false);
                    if (indexer.indexSignature != signatureFromSetFunc)
                    {
                        emitError(location) << "'set' method is not matching 'index' definition";
                        return false;
                    }

                    indexer.set = funcOp;
                }
            }

            return true;
        }

        void addAccessor()
        {
            auto &accessorInfos = newClassPtr->accessors;

            auto accessorIndex = newClassPtr->getAccessorIndex(propertyName);
            if (accessorIndex < 0)
            {
                accessorInfos.push_back({propertyName, {}, {}, isStatic, isVirtual, isAbstract});
                accessorIndex = newClassPtr->getAccessorIndex(propertyName);
            }

            assert(accessorIndex >= 0);

            if (classMember == SyntaxKind::GetAccessor)
            {
                newClassPtr->accessors[accessorIndex].get = funcOp;
            }
            else if (classMember == SyntaxKind::SetAccessor)
            {
                newClassPtr->accessors[accessorIndex].set = funcOp;
            }
        }

        ClassInfo::TypePtr newClassPtr;
        ClassElement classMember;
        std::string methodName;
        std::string propertyName;        
        bool isConstructor;
        bool isStatic;
        bool isAbstract;
        bool isExport;
        bool isImport;
        bool isForceVirtual;
        bool isVirtual;

        mlir_ts::FuncOp funcOp;
    };

    mlir::LogicalResult mlirGenClassIndexMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
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

    mlir::LogicalResult mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST,
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
        auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
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
        if (classMethodMemberInfo.isConstructor)
        {
            if (classMethodMemberInfo.isStatic && !genContext.allowPartialResolve)
            {
                createGlobalConstructor(classMember, genContext);
            }

            // adding missing statements
            generateConstructorStatements(classDeclarationAST, classMethodMemberInfo.isStatic, funcGenContext);
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

        if (newClassPtr->isPublic && hasModifier(classMember, SyntaxKind::PublicKeyword))
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
            if (classMethodMemberInfo.registerClassMethodMember(loc(funcLikeDeclaration), orderWeight))
            {
                funcLikeDeclaration->processed = true;
                return mlir::success();
            }

            return mlir::failure();
        }

        return registerGenericClassMethod(classMethodMemberInfo, genContext);
    }

    mlir::LogicalResult mlirGenClassStaticBlockMember(ClassLikeDeclaration classDeclarationAST,
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

            auto parentModule = theModule;
            MLIRCodeLogicHelper mclh(builder, location);

            builder.setInsertionPointToStart(parentModule.getBody());
            mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());            

            // priority is lowest to load as first dependencies
            builder.create<mlir_ts::GlobalConstructorOp>(
                location, mlir::FlatSymbolRefAttr::get(builder.getContext(), fullInitGlobalFuncName), builder.getIndexAttr(LAST_GLOBAL_CONSTRUCTOR_PRIORITY));            
        }

        return mlir::success();
    }

    mlir::LogicalResult registerGenericClassMethod(ClassMethodMemberInfo &classMethodMemberInfo, const GenContext &genContext)
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
                    classMethodMemberInfo.isStatic});
            }

            return mlir::success();
        }

        emitError(loc(classMethodMemberInfo.classMember)) << "virtual generic methods in class are not allowed";
        return mlir::failure();
    }

    mlir::LogicalResult mlirGenClassMethodMemberDynamicImport(ClassMethodMemberInfo &classMethodMemberInfo, int orderWeight, const GenContext &genContext)
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
        if (mlir::succeeded(mlirGenFunctionLikeDeclarationDynamicImport(
            location, funcOp.getName(), funcOp.getFunctionType(), funcOp.getName(), genContext)))
        {
            // no need to generate method in code
            funcLikeDeclaration->processed = true;
            classMethodMemberInfo.registerClassMethodMember(location, orderWeight);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult createGlobalConstructor(ClassElement classMember, const GenContext &genContext)
    {
        auto location = loc(classMember);

        auto parentModule = theModule;
        MLIRCodeLogicHelper mclh(builder, location);

        auto funcName = getNameOfFunction(classMember, genContext);

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            builder.setInsertionPointToStart(parentModule.getBody());
            mclh.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());

            builder.create<mlir_ts::GlobalConstructorOp>(location, 
                FlatSymbolRefAttr::get(builder.getContext(), StringRef(std::get<0>(funcName))), builder.getIndexAttr(LAST_GLOBAL_CONSTRUCTOR_PRIORITY));
        }

        return mlir::success();
    }

    mlir::LogicalResult generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, bool staticConstructor,
                                                      const GenContext &genContext)
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
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                auto _this = nf.createIdentifier(S(THIS_NAME));
                auto _name = nf.createIdentifier(stows(std::string(memberNamePtr)));
                auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                auto _this_name_equal = nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken),
                                                                  propertyDeclaration->initializer);
                auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
            }

            if (classMember == SyntaxKind::Constructor)
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
                        llvm_unreachable("not implemented");
                        return mlir::failure();
                    }

                    auto _this = nf.createIdentifier(stows(THIS_NAME));
                    auto _name = nf.createIdentifier(stows(std::string(propertyNamePtr)));
                    auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                    auto _this_name_equal =
                        nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), _name);
                    auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                    const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
                }
            }
        }

        return mlir::success();
    }

    bool isConstValue(Expression expr, const GenContext &genContext)
    {
        auto isConst = false;
        evaluate(
            expr, [&](mlir::Value val) { isConst = isConstValue(val); }, genContext);
        return isConst;
    }

    mlir::LogicalResult registerGenericInterface(InterfaceDeclaration interfaceDeclarationAST,
                                                 const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(interfaceDeclarationAST->name);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericInterfacesMap.count(fullNamePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(
                    processTypeParameters(interfaceDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            GenericInterfaceInfo::TypePtr newGenericInterfacePtr = std::make_shared<GenericInterfaceInfo>();
            newGenericInterfacePtr->name = namePtr;
            newGenericInterfacePtr->fullName = fullNamePtr;
            newGenericInterfacePtr->elementNamespace = currentNamespace;
            newGenericInterfacePtr->typeParams = typeParameters;
            newGenericInterfacePtr->interfaceDeclaration = interfaceDeclarationAST;

            mlirGenInterfaceType(newGenericInterfacePtr, genContext);

            getGenericInterfacesMap().insert({namePtr, newGenericInterfacePtr});
            fullNameGenericInterfacesMap.insert(fullNamePtr, newGenericInterfacePtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    void appendSpecializedTypeNames(std::string &name, NodeArray<TypeParameterDeclaration> typeParams,
                                    const GenContext &genContext)
    {
        name.append("<");
        auto next = false;
        for (auto typeParam : typeParams)
        {
            if (next)
            {
                name.append(",");
            }

            auto type = getType(typeParam, genContext);
            if (type)
            {
                llvm::raw_string_ostream s(name);
                s << type;
            }
            else
            {
                // TODO: finish it
                // name.append(MLIRHelper::getName(typeParam));
            }

            next = true;
        }

        name.append(">");
    }

    template <typename T> std::string getNameWithArguments(T declarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(declarationAST->name);
        if (name.empty())
        {
            auto [attr, result] = getNameFromComputedPropertyName(declarationAST->name, genContext);
            if (mlir::failed(result))
            {
                return nullptr;
            }

            if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(attr))
            {
                name = strAttr.getValue();
            }
        }

        if (name.empty())
        {
            if (declarationAST == SyntaxKind::ArrowFunction)
            {
                if (!genContext.receiverName.empty())
                {
                    name = genContext.receiverName.str();
                }
                else
                {
                    name = MLIRHelper::getAnonymousName(loc_check(declarationAST), ".af", "");
                }
            }
            else if (declarationAST == SyntaxKind::FunctionExpression)
            {
                name = MLIRHelper::getAnonymousName(loc_check(declarationAST), ".fe", "");
            }
            else if (declarationAST == SyntaxKind::ClassExpression)
            {
                name = MLIRHelper::getAnonymousName(loc_check(declarationAST), ".ce", "");
            }
            else if (declarationAST == SyntaxKind::Constructor)
            {
                name = CONSTRUCTOR_NAME;
            }
            else if (declarationAST == SyntaxKind::ConstructSignature)
            {
                name = NEW_CTOR_METHOD_NAME;
            }
            else
            {
                name = MLIRHelper::getAnonymousName(loc_check(declarationAST), ".unk", "");
            }
        }

        if (!name.empty() && genContext.typeParamsWithArgs.size() && declarationAST->typeParameters.size())
        {
            appendSpecializedTypeNames(name, declarationAST->typeParameters, genContext);
        }

        return name;
    }

    std::string getSpecializedInterfaceName(GenericInterfaceInfo::TypePtr geneticInterfacePtr,
                                            const GenContext &genContext)
    {
        auto name = geneticInterfacePtr->fullName.str();
        if (genContext.typeParamsWithArgs.size())
        {
            appendSpecializedTypeNames(name, geneticInterfacePtr->typeParams, genContext);
        }

        return name;
    }

    mlir_ts::InterfaceType getSpecializationInterfaceType(GenericInterfaceInfo::TypePtr genericInterfacePtr,
                                                          const GenContext &genContext)
    {
        auto fullSpecializedInterfaceName = getSpecializedInterfaceName(genericInterfacePtr, genContext);
        auto interfaceInfoType = getInterfaceInfoByFullName(fullSpecializedInterfaceName);
        assert(interfaceInfoType);
        interfaceInfoType->originInterfaceType = genericInterfacePtr->interfaceType;
        return interfaceInfoType->interfaceType;
    }

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(InterfaceDeclaration interfaceDeclarationAST, bool &declareInterface,
                                                const GenContext &genContext)
    {
        auto name = getNameWithArguments(interfaceDeclarationAST, genContext);
        return mlirGenInterfaceInfo(name, declareInterface, genContext);
    }

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(const std::string &name, bool &declareInterface,
                                                const GenContext &genContext)
    {
        declareInterface = false;

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        InterfaceInfo::TypePtr newInterfacePtr;
        if (fullNameInterfacesMap.count(fullNamePtr))
        {
            newInterfacePtr = fullNameInterfacesMap.lookup(fullNamePtr);
            getInterfacesMap().insert({namePtr, newInterfacePtr});
            declareInterface = !newInterfacePtr->interfaceType;
        }
        else
        {
            // register class
            newInterfacePtr = std::make_shared<InterfaceInfo>();
            newInterfacePtr->name = namePtr;
            newInterfacePtr->fullName = fullNamePtr;
            newInterfacePtr->elementNamespace = currentNamespace;

            getInterfacesMap().insert({namePtr, newInterfacePtr});
            fullNameInterfacesMap.insert(fullNamePtr, newInterfacePtr);
            declareInterface = true;
        }

        if (declareInterface && mlir::succeeded(mlirGenInterfaceType(newInterfacePtr, genContext)))
        {
            newInterfacePtr->typeParamsWithArgs = genContext.typeParamsWithArgs;
        }

        return newInterfacePtr;
    }

    mlir::LogicalResult mlirGenInterfaceHeritageClauseExtends(InterfaceDeclaration interfaceDeclarationAST,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              HeritageClause heritageClause, int &orderWeight, bool declareClass,
                                                              const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ExtendsKeyword)
        {
            return mlir::success();
        }

        for (auto &extendsType : heritageClause->types)
        {
            auto result = mlirGen(extendsType, genContext);
            EXIT_IF_FAILED(result);
            auto ifaceType = V(result);
            auto success = false;
            mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    if (interfaceInfo)
                    {
                        newInterfacePtr->extends.push_back({-1, interfaceInfo});
                        success = true;
                    }
                })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) {
                    llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
                    if (mlir::succeeded(mth.getFields(tupleType, destTupleFields)))
                    {
                        orderWeight++;
                        success = true;
                        for (auto field : destTupleFields)
                            success &= mlir::succeeded(
                                mlirGenInterfaceAddFieldMember(newInterfacePtr, field.id, field.type, field.isConditional, orderWeight));
                    }
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        if (interfaceDeclarationAST->typeParameters.size() > 0 && genContext.typeParamsWithArgs.size() == 0)
        {
            return registerGenericInterface(interfaceDeclarationAST, genContext);
        }

        auto declareInterface = false;
        auto newInterfacePtr = mlirGenInterfaceInfo(interfaceDeclarationAST, declareInterface, genContext);
        if (!newInterfacePtr)
        {
            return mlir::failure();
        }

        // do not process specialized interface second time;
        if (!declareInterface && interfaceDeclarationAST->typeParameters.size() > 0 &&
            genContext.typeParamsWithArgs.size() > 0)
        {
            return mlir::success();
        }

        auto location = loc(interfaceDeclarationAST);

        auto ifaceGenContext = GenContext(genContext);
        ifaceGenContext.thisType = newInterfacePtr->interfaceType;

        auto orderWeight = 0;
        for (auto &heritageClause : interfaceDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenInterfaceHeritageClauseExtends(interfaceDeclarationAST, newInterfacePtr,
                                                                   heritageClause, orderWeight, declareInterface, genContext)))
            {
                return mlir::failure();
            }
        }

        newInterfacePtr->recalcOffsets();

        // clear all flags
        for (auto &interfaceMember : interfaceDeclarationAST->members)
        {
            interfaceMember->processed = false;
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            for (auto &interfaceMember : interfaceDeclarationAST->members)
            {
                orderWeight++;
                if (mlir::failed(mlirGenInterfaceMethodMember(
                        interfaceDeclarationAST, newInterfacePtr, interfaceMember, orderWeight, declareInterface, ifaceGenContext)))
                {
                    notResolved++;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // interface can depend on other interface declarations
                // theModule.emitError("can't resolve dependencies in intrerface: ") << newInterfacePtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        // add to export if any
        if (auto hasExport = getExportModifier(interfaceDeclarationAST))
        {
            addInterfaceDeclarationToExport(newInterfacePtr);
        }

        return mlir::success();
    }

    template <typename T> mlir::LogicalResult mlirGenInterfaceType(T newInterfacePtr, const GenContext &genContext)
    {
        if (newInterfacePtr)
        {
            newInterfacePtr->interfaceType = getInterfaceType(newInterfacePtr->fullName);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenInterfaceAddFieldMember(InterfaceInfo::TypePtr newInterfacePtr, mlir::Attribute fieldId, mlir::Type typeIn, bool isConditional, int orderWeight, bool declareInterface = true)
    {
        auto &fieldInfos = newInterfacePtr->fields;
        auto type = typeIn;

        // fix type for fields with FuncType
        if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
        {

            auto funcType = getFunctionType(hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg());
            type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
        }
        else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
        {

            type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
        }

        if (mth.isNoneType(type))
        {
            LLVM_DEBUG(dbgs() << "\n!! interface field: " << fieldId << " FAILED\n");
            return mlir::failure();
        }

        auto fieldIndex = newInterfacePtr->getFieldIndex(fieldId);
        if (fieldIndex == -1)
        {
            fieldInfos.push_back({fieldId, type, isConditional, orderWeight, newInterfacePtr->getNextVTableMemberIndex()});
        }
        else
        {
            // update
            fieldInfos[fieldIndex].type = type;
            fieldInfos[fieldIndex].isConditional = isConditional;
        }

        return mlir::success();
    }

    mlir::LogicalResult addInterfaceMethod(mlir::Location location, InterfaceInfo::TypePtr newInterfacePtr,
        llvm::SmallVector<InterfaceMethodInfo> &methodInfos, StringRef methodName, mlir_ts::FunctionType funcType, bool isConditional,
        int orderWeight, bool declareInterface, const GenContext &genContext)
    {
        if (methodName.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::failure();
        }

        if (!funcType)
        {
            return mlir::failure();
        }

        if (llvm::any_of(funcType.getInputs(), [&](mlir::Type type) { return !type; }))
        {
            return mlir::failure();
        }

        if (llvm::any_of(funcType.getResults(), [&](mlir::Type type) { return !type; }))
        {
            return mlir::failure();
        }

        auto methodIndex = newInterfacePtr->getMethodIndex(methodName);
        if (methodIndex == -1)
        {
            methodInfos.push_back(
                {methodName.str(), funcType, isConditional, orderWeight, newInterfacePtr->getNextVTableMemberIndex()});
        }
        else
        {
            methodInfos[methodIndex].funcType = funcType;
            methodInfos[methodIndex].isConditional = isConditional;
        }

        return mlir::success();    
    }

    mlir::LogicalResult getInterfaceMethodNameAndType(mlir::Location location, mlir_ts::InterfaceType interfaceType,
        MethodSignature methodSignature, std::string &methodNameOut, std::string &propertyNameOut, mlir_ts::FunctionType &funcTypeOut, 
        const GenContext &genContext) {

        std::string methodName;
        std::string propertyName;
        getMethodNameOrPropertyName(false, methodSignature, methodName, propertyName, genContext);

        methodNameOut = methodName;
        propertyNameOut = propertyName;
        
        if (methodSignature->typeParameters.size() > 0)
        {
            emitError(location) << "Generic method '" << methodName << "' in the interface is not allowed";
            return mlir::failure();
        }

        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        funcGenContext.thisType = interfaceType;

        auto res = mlirGenFunctionSignaturePrototype(methodSignature, true, funcGenContext);
        auto funcType = std::get<1>(res);
        funcTypeOut = funcType;

        return mlir::success();
    }

    mlir::LogicalResult mlirGenInterfaceMethodMember(InterfaceDeclaration interfaceDeclarationAST,
                                                     InterfaceInfo::TypePtr newInterfacePtr,
                                                     TypeElement interfaceMember, int orderWeight, bool declareInterface,
                                                     const GenContext &genContext)
    {
        if (interfaceMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(interfaceMember);

        auto &methodInfos = newInterfacePtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        MLIRCodeLogic mcl(builder);

        SyntaxKind kind = interfaceMember;
        if (kind == SyntaxKind::PropertySignature)
        {
            // property declaration
            auto propertySignature = interfaceMember.as<PropertySignature>();
            auto isConditional = !!propertySignature->questionToken;

            fieldId = TupleFieldName(propertySignature->name, genContext);

            auto [type, init, typeProvided] = getTypeAndInit(propertySignature, genContext);
            if (!type)
            {
                return mlir::failure();
            }

            if (mlir::failed(mlirGenInterfaceAddFieldMember(newInterfacePtr, fieldId, type, isConditional, orderWeight, declareInterface)))
            {
                return mlir::failure();
            }
        }
        else if (kind == SyntaxKind::MethodSignature 
                || kind == SyntaxKind::ConstructSignature || kind == SyntaxKind::CallSignature 
                || kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            auto isConditional = !!methodSignature->questionToken;

            newInterfacePtr->hasNew |= kind == SyntaxKind::ConstructSignature;
            // we need this code to add "THIS" param to declaration
            interfaceMember->parent = interfaceDeclarationAST;

            std::string methodName;
            std::string propertyName;
            mlir_ts::FunctionType funcType;
            if (mlir::failed(getInterfaceMethodNameAndType(location, newInterfacePtr->interfaceType, methodSignature, 
                    methodName, propertyName, funcType, genContext)))
            {
                return mlir::failure();
            }

            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                methodName, funcType, isConditional, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            // add info about property
            if (kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor)
            {
                auto accessor = newInterfacePtr->findAccessor(propertyName);
                
                auto &accessors = newInterfacePtr->accessors;
                if (accessor == nullptr)
                {
                    if (kind == SyntaxKind::GetAccessor)
                    {
                        accessors.push_back({funcType.getResult(0), propertyName, methodName, ""});
                    }
                    else
                    {
                        accessors.push_back({funcType.getInputs().back(), propertyName, "", methodName});
                    }
                }
                else
                {
                    if (kind == SyntaxKind::GetAccessor)
                    {
                        accessor->getMethod = methodName;
                    }
                    else
                    {
                        accessor->setMethod = methodName;
                    }                    
                }
            }

            methodSignature->processed = true;
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            // we need this code to add "THIS" param to declaration
            interfaceMember->parent = interfaceDeclarationAST;

            std::string methodName;
            std::string propertyName;
            mlir_ts::FunctionType funcType;
            if (mlir::failed(getInterfaceMethodNameAndType(
                location, newInterfacePtr->interfaceType, methodSignature, methodName, propertyName, funcType, genContext)))
            {
                return mlir::failure();
            }

            // add get method
            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                INDEX_ACCESS_GET_FIELD_NAME, mth.getIndexGetFunctionType(funcType), true, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                INDEX_ACCESS_SET_FIELD_NAME, mth.getIndexSetFunctionType(funcType), true, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            auto found = llvm::find_if(newInterfacePtr->indexes, [&] (auto indexInfo) {
                return indexInfo.indexSignature == funcType;
            });        

            if (found == newInterfacePtr->indexes.end())
            {
                newInterfacePtr->indexes.push_back({funcType, INDEX_ACCESS_GET_FIELD_NAME, INDEX_ACCESS_SET_FIELD_NAME});
            }   

            methodSignature->processed = true;            
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::success();
    }

    std::tuple<std::string, bool> getNameForMethod(SignatureDeclarationBase methodSignature, const GenContext &genContext)
    {
        auto [attr, result] = getNameFromComputedPropertyName(methodSignature->name, genContext);
        if (mlir::failed(result))
        {
            return {"", false};
        }

        if (attr)
        {
            if (auto strAttr = dyn_cast<mlir::StringAttr>(attr))
            {
                return {strAttr.getValue().str(), true};
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return {MLIRHelper::getName(methodSignature->name), true};
    }

    mlir::LogicalResult getMethodNameOrPropertyName(bool isStaticClass, SignatureDeclarationBase methodSignature, std::string &methodName,
                                                    std::string &propertyName, const GenContext &genContext)
    {
        SyntaxKind kind = methodSignature;
        if (kind == SyntaxKind::Constructor)
        {
            auto isStatic = isStaticClass || hasModifier(methodSignature, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                methodName = std::string(STATIC_CONSTRUCTOR_NAME);
            }
            else
            {
                methodName = std::string(CONSTRUCTOR_NAME);
            }
        }
        else if (kind == SyntaxKind::ConstructSignature)
        {
            methodName = std::string(NEW_CTOR_METHOD_NAME);
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            methodName = std::string(INDEX_ACCESS_FIELD_NAME);
        }
        else if (kind == SyntaxKind::CallSignature)
        {
            methodName = std::string(CALL_FIELD_NAME);
        }
        else if (kind == SyntaxKind::GetAccessor)
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }

            propertyName = name;
            methodName = std::string("get_") + propertyName;
        }
        else if (kind == SyntaxKind::SetAccessor)
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }

            propertyName = name;
            methodName = std::string("set_") + propertyName;
        }
        else
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }            

            methodName = name;
        }

        return mlir::success();
    }

    mlir::Block* prepareTempModule()
    {
        if (tempEntryBlock)
        {
            theModule = tempModule;
            return tempEntryBlock;
        }

        auto location = loc(TextRange());

        theModule = tempModule = mlir::ModuleOp::create(location, mlir::StringRef("temp_module"));

        // we need to add temporary block
        auto tempFuncType =
            mlir_ts::FunctionType::get(builder.getContext(), ArrayRef<mlir::Type>(), ArrayRef<mlir::Type>());
        tempFuncOp = mlir_ts::FuncOp::create(location, ".tempfunc", tempFuncType);

        tempEntryBlock = tempFuncOp.addEntryBlock();

        return tempEntryBlock;
    }

    void clearTempModule()
    {
        if (tempEntryBlock)
        {
            tempEntryBlock->dropAllDefinedValueUses();
            tempEntryBlock->dropAllUses();
            tempEntryBlock->dropAllReferences();
            tempEntryBlock->erase();

            tempFuncOp.erase();
            tempModule.erase();

            tempEntryBlock = nullptr;
        }
    }

    mlir::Type evaluate(Expression expr, const GenContext &genContext)
    {
        // we need to add temporary block
        mlir::Type result;
        if (expr)
        {
            evaluate(
                expr, [&](mlir::Value val) { result = val.getType(); }, genContext);
        }

        return result;
    }

    void evaluate(Expression expr, std::function<void(mlir::Value)> func, const GenContext &genContext)
    {
        if (!expr)
        {
            return;
        }

        // TODO: sometimes we need errors, sometimes, not,
        // we need to ignore errors;
        //mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
        //});

        auto location = loc(expr);

        // module
        auto savedModule = theModule;

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            SymbolTableScopeT varScope(symbolTable);

            builder.setInsertionPointToStart(prepareTempModule());

            GenContext evalGenContext(genContext);
            evalGenContext.allowPartialResolve = true;
            evalGenContext.funcOp = tempFuncOp;
            auto result = mlirGen(expr, evalGenContext);
            auto initValue = V(result);
            if (initValue)
            {
                func(initValue);
            }
        }

        theModule = savedModule;
    }

    mlir::Value evaluatePropertyValue(mlir::Location location, mlir::Value exprValue, const std::string &propertyName, const GenContext &genContext)
    {
        // we need to ignore errors;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
        });

        mlir::Value initValue;

        // module
        auto savedModule = theModule;

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(prepareTempModule());

            GenContext evalGenContext(genContext);
            evalGenContext.allowPartialResolve = true;
            evalGenContext.funcOp = tempFuncOp;
            auto result = mlirGenPropertyAccessExpression(location, exprValue, propertyName, evalGenContext);
            initValue = V(result);
        }

        theModule = savedModule;

        return initValue;
    }    

    // TODO: rewrite code to get rid of the following method, write method to calculate type of field, we have method mth.getFieldTypeByFieldName
    mlir::Type evaluateProperty(mlir::Location location, mlir::Value exprValue, const std::string &propertyName, const GenContext &genContext)
    {
        auto value = evaluatePropertyValue(location, exprValue, propertyName, genContext);
        return value ? value.getType() : mlir::Type();
    }

    mlir::Type evaluateProperty(Expression expression, const std::string &propertyName, const GenContext &genContext)
    {
        auto location = loc(expression);

        auto result = mlirGen(expression, genContext);
        if (result.failed_or_no_value())
        {
            return mlir::Type();
        }

        auto exprValue = V(result);

        auto value = evaluatePropertyValue(location, exprValue, propertyName, genContext);
        return value ? value.getType() : mlir::Type();
    }

    mlir::Type evaluateElementAccess(mlir::Location location, mlir::Value expression, bool isConditionalAccess, const GenContext &genContext)
    {
        // we need to ignore errors;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
        });

        mlir::Type resultType;

        // module
        auto savedModule = theModule;

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(prepareTempModule());

            GenContext evalGenContext(genContext);
            evalGenContext.allowPartialResolve = true;
            auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getStructIndexType(),
                                    mth.getStructIndexAttrValue(0));
            auto result = mlirGenElementAccess(location, expression, indexVal, isConditionalAccess, evalGenContext);
            auto initValue = V(result);
            if (initValue)
            {
                resultType = initValue.getType();
            }
        }

        theModule = savedModule;

        return resultType;
    }    

    ValueOrLogicalResult mapTupleToFields(mlir::Location location, SmallVector<mlir::Value> &values, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, bool filterSpecialCases, const GenContext &genContext, bool errorAsWarning = false)
    {
        auto count = 0;
        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);           

            if (filterSpecialCases)
            {
                // filter out special fields
                if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fieldInfo.id)) 
                {
                    if (strAttr.getValue().starts_with(".")) {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! --filtered #" << index << " field [" << fieldInfo.id << "]\n";);           
                        continue;
                    }
                }
            }

            count ++;
            if (fieldInfo.id == mlir::Attribute() || (index < srcTupleType.size() && srcTupleType.getFieldInfo(index).id == mlir::Attribute()))
            {
                if (index >= srcTupleType.size() && isa<mlir_ts::OptionalType>(fieldInfo.type))
                {
                    // add undefined value
                    auto undefVal = builder.create<mlir_ts::OptionalUndefOp>(location, fieldInfo.type);
                    values.push_back(undefVal);
                    continue;
                }

                MLIRPropertyAccessCodeLogic cl(builder, location, value, builder.getI32IntegerAttr(index));
                auto value = cl.Tuple(srcTupleType, true);
                VALIDATE(value, location)
                values.push_back(value);
            }
            else
            {
                // access by field name
                auto fieldIndex = srcTupleType.getIndex(fieldInfo.id);
                if (fieldIndex < 0)
                {
                    if (isa<mlir_ts::OptionalType>(fieldInfo.type))
                    {
                        // add undefined value
                        auto undefVal = builder.create<mlir_ts::OptionalUndefOp>(location, fieldInfo.type);
                        values.push_back(undefVal);
                        continue;
                    }

                    if (errorAsWarning)
                    {
                        emitWarning(location)
                            << "field " << fieldInfo.id << " can't be found in tuple '" << srcTupleType << "'";

                        // add undefined value
                        auto undefVal = builder.create<mlir_ts::UndefOp>(location, fieldInfo.type);
                        values.push_back(undefVal);
                        continue;
                    }
                    
                    emitError(location)
                        << "field " << fieldInfo.id << " can't be found in tuple '" << to_print(srcTupleType) << "'";
                    return mlir::failure();
                }                

                MLIRPropertyAccessCodeLogic cl(builder, location, value, fieldInfo.id);
                // TODO: implement conditional
                auto propertyAccess = mlirGenPropertyAccessExpressionLogic(location, value, false, cl, genContext); 
                EXIT_IF_FAILED_OR_NO_VALUE(propertyAccess)

                auto value = V(propertyAccess);
                if (value.getType() != fieldInfo.type)
                {
                    CAST(value, location, fieldInfo.type, value, genContext)
                }

                values.push_back(value);
            }
        }

        if (count != values.size())
        {
            emitError(location)
                << "count of fields (" << count << ") in destination is not matching to " << to_print(srcTupleType) << "'";            
            return mlir::failure();
        }

        return mlir::success();
    }    


    ValueOrLogicalResult castTupleToTuple(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext, bool errorAsWarning = false)
    {
        SmallVector<mlir::Value> values;

        auto result = mapTupleToFields(location, values, value, srcTupleType, fields, false, genContext, errorAsWarning);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }    

    ValueOrLogicalResult castTupleToClass(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, mlir_ts::ClassType classType, const GenContext &genContext, bool errorAsWarning = false)
    {
        SmallVector<mlir::Value> values;
        
        auto result = mapTupleToFields(location, values, value, srcTupleType, fields, true, genContext, errorAsWarning);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        // SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        // fieldsForTuple.append(fields.begin(), fields.end());
        // return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));

        SmallVector<mlir::Value, 4> operands;
        auto newInstanceOfClass = NewClassInstance(location, classType, operands, genContext);
        // TODO: assign fields to values
        
        auto valueIndex = 0;
        for (auto fieldInfo : fields)
        {
            // filter out special fields
            if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fieldInfo.id)) 
            {
                if (strAttr.getValue().starts_with(".")) {
                    continue;
                }
            }            

            auto value = values[valueIndex];

            MLIRPropertyAccessCodeLogic cl(builder, location, newInstanceOfClass, fieldInfo.id);
            // TODO: implement conditional
            auto propertyAccess = mlirGenPropertyAccessExpressionLogic(location, newInstanceOfClass, false, cl, genContext); 
            EXIT_IF_FAILED_OR_NO_VALUE(propertyAccess)

            auto property = V(propertyAccess);
            if (value.getType() != fieldInfo.type)
            {
                CAST(value, location, fieldInfo.type, value, genContext)
            }

            auto result = mlirGenSaveLogicOneItem(location, property, value, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)

            valueIndex++;
        }

        return newInstanceOfClass;
    }    

    // TODO: finish it
    ValueOrLogicalResult castConstArrayToString(mlir::Location location, mlir::Value value, const GenContext &genContext)
    {
        if (auto constArray = dyn_cast<mlir_ts::ConstArrayType>(value.getType()))
        {
            auto stringType = getStringType();
            SmallVector<mlir::Value, 4> strs;

            auto spaceText = " ";
            auto spaceValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spaceText));

            auto spanText = ",";
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            auto beginText = "[";
            auto beginValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(beginText));

            auto endText = "]";
            auto endValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(endText));

            strs.push_back(beginValue);

            auto constantOp = value.getDefiningOp<mlir_ts::ConstantOp>();
            auto arrayAttr = mlir::cast<mlir::ArrayAttr>(constantOp.getValue());
            for (auto [index, val] : enumerate(arrayAttr))
            {
                if (index > 0) 
                {
                    // text
                    strs.push_back(spanValue);
                }

                // we need to convert it into string
                if (auto typedAttr = dyn_cast<mlir::TypedAttr>(val))
                {
                    strs.push_back(spaceValue);

                    auto itemConstValue = builder.create<mlir_ts::ConstantOp>(location, typedAttr);
                    if (itemConstValue.getType() != stringType)
                    {
                        CAST_A(convertedValue, location, stringType, itemConstValue, genContext);
                        strs.push_back(convertedValue);
                    }
                    else
                    {
                        strs.push_back(itemConstValue);                
                    }                    
                }
                else
                {
                    return mlir::failure();
                }
            }

            if (strs.size() > 1)
            {
                strs.push_back(spaceValue);
            }

            strs.push_back(endValue);

            if (strs.size() <= 0)
            {
                return V(builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr("")));
            }

            auto concatValues =
                builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

            return V(concatValues);    
        }    

        return mlir::failure();
    }     

    ValueOrLogicalResult castTupleToString(mlir::Location location, mlir::Value value, mlir_ts::TupleType tupleType,
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto spaceText = " ";
        auto spaceValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spaceText));

        auto fieldSepText = ": ";
        auto fieldSepValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(fieldSepText));

        auto spanText = ",";
        auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

        auto quotText = "'";
        auto quotValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(quotText));

        auto beginText = "{";
        auto beginValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(beginText));

        auto endText = "}";
        auto endValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(endText));

        strs.push_back(beginValue);

        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);           

            if (index > 1) 
            {
                // text
                strs.push_back(spanValue);
            }

            strs.push_back(spaceValue);
            if (fieldInfo.id)
            {
                auto fieldNameValue = builder.create<mlir_ts::ConstantOp>(location, stringType, fieldInfo.id);
                strs.push_back(fieldNameValue);
                strs.push_back(fieldSepValue);
            }

            MLIRPropertyAccessCodeLogic cl(builder, location, value, builder.getI32IntegerAttr(index));
            auto fieldValue = cl.Tuple(tupleType, true);
            VALIDATE(value, location)

            if (fieldValue.getType() != stringType)
            {
                CAST(fieldValue, location, stringType, fieldValue, genContext);
                // expr value
                strs.push_back(fieldValue);
            }
            else
            {
                // expr value
                strs.push_back(quotValue);
                strs.push_back(fieldValue);
                strs.push_back(quotValue);
            }
        }

        if (strs.size() > 1)
        {
            strs.push_back(spaceValue);
        }

        strs.push_back(endValue);

        if (strs.size() <= 0)
        {
            return V(builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr("")));
        }

        auto concatValues =
            builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return V(concatValues);        
    }       

    ValueOrLogicalResult generatingStaticNewCtorForClass(mlir::Location location, ClassInfo::TypePtr classInfo, int posIndex, const GenContext &genContext)
    {
        if (auto classConstrMethodInfo = classInfo->findMethod(CONSTRUCTOR_NAME))
        {
            auto funcWithReturnClass = getFunctionType(
                classConstrMethodInfo->funcType.getInputs().slice(1) /*to remove this*/, 
                {classInfo->classType}, 
                classConstrMethodInfo->funcType.isVarArg());
            auto foundNewCtoreStaticMethodFullName = generateSynthMethodToCallNewCtor(location, classInfo, classInfo->fullName, posIndex, funcWithReturnClass, 0, genContext);
            if (foundNewCtoreStaticMethodFullName.empty())
            {
                return mlir::failure();
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, funcWithReturnClass,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), foundNewCtoreStaticMethodFullName));
        
            return V(symbOp);
        }
        else
        {
            emitError(location) << "constructor can't be found";
            return mlir::failure();
        }
    }

    ValueOrLogicalResult castClassToTuple(mlir::Location location, mlir::Value value, mlir_ts::ClassType classType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        assert(classInfo);            

        auto newCtorAttr = MLIRHelper::TupleFieldName(NEW_CTOR_METHOD_NAME, builder.getContext());
        SmallVector<mlir::Value> values;
        for (auto [posIndex, fieldInfo] : enumerate(fields))
        {
            auto foundField = false;                                        
            auto classFieldInfo = classInfo->findField(fieldInfo.id, foundField);
            if (!foundField)
            {
                // TODO: generate method wrapper for calling new/ctor method
                if (fieldInfo.id == newCtorAttr)
                {
                    auto newCtorSymbOp = generatingStaticNewCtorForClass(location, classInfo, posIndex, genContext);
                    EXIT_IF_FAILED_OR_NO_VALUE(newCtorSymbOp)
                    values.push_back(newCtorSymbOp);
                    continue;
                }

                emitError(location)
                    << "field " << fieldInfo.id << " can't be found in class '" << classInfo->fullName << "'";
                return mlir::failure();
            }                

            MLIRPropertyAccessCodeLogic cl(builder, location, value, fieldInfo.id);
            // TODO: implemenet conditional
            mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, value, false, cl, genContext); 
            if (propertyAccess)
            {
                values.push_back(propertyAccess);
            }
        }

        if (fields.size() != values.size())
        {
            return mlir::failure();
        }        

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }

    ValueOrLogicalResult castInterfaceToTuple(mlir::Location location, mlir::Value value, mlir_ts::InterfaceType interfaceType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        assert(interfaceInfo);            

        SmallVector<mlir::Value> values;
        for (auto fieldInfo : fields)
        {
            auto classFieldInfo = interfaceInfo->findField(fieldInfo.id);
            if (!classFieldInfo)
            {
                emitError(location)
                    << "field '" << fieldInfo.id << "' can't be found "
                    << "' in interface '" << interfaceInfo->fullName << "'";
                return mlir::failure();
            }                

            MLIRPropertyAccessCodeLogic cl(builder, location, value, fieldInfo.id);
            // TODO: implemenet conditional
            mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(
                location, value, classFieldInfo->isConditional, cl, genContext); 
            if (propertyAccess)
            {
                values.push_back(propertyAccess);
            }
        }

        if (fields.size() != values.size())
        {
            return mlir::failure();
        }          

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }

    // TODO: cast should not throw error in case of generic methods in "if (false)" conditions (typeof == "..."), 
    // as it may prevent cmpiling code
    ValueOrLogicalResult cast(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        if (!type)
        {
            return mlir::failure();
        }

        if (type == value.getType())
        {
            return value;
        }

        auto valueType = value.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! cast " << valueType << " -> " << type
                                << "\n";);

        if (auto litType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            if (auto valLitType = dyn_cast<mlir_ts::LiteralType>(valueType))
            {
                if (litType.getValue() != valLitType.getValue())
                {
                    emitError(location, "can't cast from literal type: '") << valLitType.getValue() << "' to '" << litType.getValue() << "'";
                    return mlir::failure(); 
                }
            }
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(valueType))
        {
            value = builder.create<mlir_ts::CastOp>(location, enumType.getElementType(), value);
            valueType = value.getType();
        }        

        // toPrimitive
        if ((isa<mlir_ts::StringType>(type) 
            || isa<mlir_ts::NumberType>(type) 
            || isa<mlir_ts::BigIntType>(type) 
            || isa<mlir_ts::BooleanType>(type) 
            || isa<mlir_ts::UndefinedType>(type) 
            || isa<mlir_ts::SymbolType>(type) 
            || isa<mlir_ts::NullType>(type))
            && (isa<mlir_ts::ClassType>(valueType)
                || isa<mlir_ts::ClassStorageType>(valueType)
                || isa<mlir_ts::ObjectType>(valueType)
                || isa<mlir_ts::InterfaceType>(valueType)
                || isa<mlir_ts::TupleType>(valueType)
                || isa<mlir_ts::ConstTupleType>(valueType)))
        {
            // check if we need to call toPrimitive
            if (auto toPrimitiveType = evaluateProperty(location, value, SYMBOL_TO_PRIMITIVE, genContext))
            {
                NodeFactory nf(NodeFactoryFlags::None);
                Expression hint;

                mlir::TypeSwitch<mlir::Type>(type)
                    .template Case<mlir_ts::StringType>([&](auto) {
                        hint = nf.createStringLiteral(S("string"));
                    })
                    .template Case<mlir_ts::NumberType>([&](auto) {
                        hint = nf.createStringLiteral(S("number"));
                    })
                    .template Case<mlir_ts::BigIntType>([&](auto) {
                        hint = nf.createStringLiteral(S("bigint"));
                    })
                    .template Case<mlir_ts::BooleanType>([&](auto) {
                        hint = nf.createStringLiteral(S("boolean"));
                    })
                    .template Case<mlir_ts::UndefinedType>([&](auto) {
                        hint = nf.createStringLiteral(S(UNDEFINED_NAME));
                    })
                    .template Case<mlir_ts::SymbolType>([&](auto) {
                        hint = nf.createStringLiteral(S("symbol"));
                    })
                    .template Case<mlir_ts::NullType>([&](auto) {
                        hint = nf.createStringLiteral(S("null"));
                    })
                    .Default([&](auto type) {});

                auto callResult = mlirGenCallThisMethod(location, value, SYMBOL_TO_PRIMITIVE, undefined, {hint}, genContext);
                EXIT_IF_FAILED(callResult);
                auto callResultValue = V(callResult);
                if (isa<mlir_ts::UnionType>(callResultValue.getType()))
                {
                    return V(builder.create<mlir_ts::GetValueFromUnionOp>(location, type, callResultValue));                    
                }

                auto castValue = cast(location, type, callResultValue, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(castValue);
                return castValue;
            } 
        }

        // class or array or tuple to string
        if (auto stringType = dyn_cast<mlir_ts::StringType>(type))
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto res = mlirGenCallThisMethod(location, value, "get_" SYMBOL_TO_STRING_TAG, undefined, undefined, genContext);
                if (!res.failed_or_no_value())
                {
                    return res;
                }
                
                return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
            }

            if (auto arrayType = dyn_cast<mlir_ts::ConstArrayType>(valueType))
            {
                return castConstArrayToString(location, value, genContext);
            }
            else if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(valueType))
            {
                // we evaluate property to allow to compile code in "generic methods" with "typeof" conditions
                // if we throw error here generic method with "if (false)" condition will generate code which
                // will be removed but because of error, the compilation process will be stopped
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }
            }

            if (auto srcConstTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
            {
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }

                return castTupleToString(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), 
                    srcConstTupleType.getFields(), genContext);
            }
            else if (auto srcTupleType = dyn_cast<mlir_ts::TupleType>(valueType))
            {
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }

                return castTupleToString(location, value, srcTupleType, srcTupleType.getFields(), genContext);
            }
        }

        // <???> to interface
        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto result = mlirGenPropertyAccessExpression(location, value, VTABLE_NAME, genContext);
                auto vtableAccess = V(result);

                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);

                auto implementIndex = classInfo->getImplementIndex(interfaceType.getName().getValue());
                if (implementIndex >= 0)
                {
                    auto interfaceVirtTableIndex = classInfo->implements[implementIndex].virtualIndex;

                    assert(genContext.allowPartialResolve || interfaceVirtTableIndex >= 0);

                    auto interfaceVTablePtr = builder.create<mlir_ts::VTableOffsetRefOp>(
                        location, mth.getInterfaceVTableType(interfaceType), vtableAccess, interfaceVirtTableIndex);

                    auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(
                        location, mlir::TypeRange{interfaceType}, value, interfaceVTablePtr);
                    return V(newInterface);
                }

                // create interface vtable from current class
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                assert(interfaceInfo);

                if (auto createdInterfaceVTableForClass =
                        mlirGenCreateInterfaceVTableForClass(location, classInfo, interfaceInfo, genContext))
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                            << "@ created interface:" << V(createdInterfaceVTableForClass) << "\n";);
                    auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(
                        location, mlir::TypeRange{interfaceType}, value, createdInterfaceVTableForClass);

                    return V(newInterface);
                }

                emitError(location) << "type: " << classType.getName() << " missing interface: " << interfaceType.getName();
                return mlir::failure();
            }

            // tuple to interface
            if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
            {
                return castTupleToInterface(location, value, constTupleType, interfaceType, genContext);
            }

            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(valueType))
            {
                return castTupleToInterface(location, value, tupleType, interfaceType, genContext);
            }

            // object to interface
            if (auto objectType = dyn_cast<mlir_ts::ObjectType>(valueType))
            {
                return castObjectToInterface(location, value, objectType, interfaceType, genContext);
            }
        }

        // const tuple to tuple
        if (auto srcConstTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castTupleToTuple(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castTupleToTuple(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, genContext);
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                fields = mlir::cast<mlir_ts::ClassStorageType>(classType.getStorageType()).getFields();     
                return castTupleToClass(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, classType, genContext);                
            }
            else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }            
        }

        // tuple to tuple
        if (auto srcTupleType = dyn_cast<mlir_ts::TupleType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castTupleToTuple(location, value, srcTupleType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castTupleToTuple(location, value, srcTupleType, fields, genContext);
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
            else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }               
        }

        // class to tuple
        if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castClassToTuple(location, value, classType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castClassToTuple(location, value, classType, fields, genContext);
            }
        }

        // interface to tuple
        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castInterfaceToTuple(location, value, interfaceType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castInterfaceToTuple(location, value, interfaceType, fields, genContext);
            }
        }

        // optional
        // TODO: it is in CastLogic as well, review usage and remove from here
        // but if optional points to interface then it will not work
        // example: from path.ts
        // %6 = ts.Cast %4 : !ts.const_tuple<{"key",!ts.string},{"prev",!ts.undefined},{"typename",!ts.undefined}> to !ts.optional<!ts.iface<@Path>>
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            if (valueType == getUndefinedType())
            {
                return V(builder.create<mlir_ts::OptionalUndefOp>(location, optType));
            }
            else if (auto optValueType = dyn_cast<mlir_ts::OptionalType>(valueType))
            {
                auto condValue = builder.create<mlir_ts::HasValueOp>(location, getBooleanType(), value);
                return optionalValueOrUndefined(
                    location, 
                    condValue, 
                    [&](auto genContext) 
                    { 
                        auto valueFromOptional = builder.create<mlir_ts::ValueOp>(location, optValueType.getElementType(), value);
                        return cast(location, optType.getElementType(), valueFromOptional, genContext);
                    }, 
                    genContext);
            }
            else
            {
                CAST_A(valueCasted, location, optType.getElementType(), value, genContext);
                return V(builder.create<mlir_ts::OptionalValueOp>(location, optType, valueCasted));
            }
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                auto types = unionType.getTypes();
                if (std::find(types.begin(), types.end(), valueType) == types.end())
                {
                    // find which type we can cast to
                    for (auto subType : types)
                    {
                        if (mth.canCastFromTo(location, valueType, subType))
                        {
                            CAST(value, location, subType, value, genContext);
                            return V(builder.create<mlir_ts::CastOp>(location, type, value));                    
                        }
                    }
                }
                else
                {
                    return V(builder.create<mlir_ts::CastOp>(location, type, value));                    
                }
            }
        }

        if (auto constType = dyn_cast<mlir_ts::ConstType>(type))
        {
            // TODO: we can't convert array to const array

            auto currType = valueType;
            if (auto refType = dyn_cast<mlir_ts::RefType>(currType))
            {
                type = refType.getElementType();        
            }
            else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(currType))
            {
                type = mth.convertTupleTypeToConstTupleType(tupleType);                
            }
            else
            {
                return value;
            }
        }

        // union type to <basic type>
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(valueType))
        {
            // union -> any will be done later in CastLogic
            auto toAny = dyn_cast<mlir_ts::AnyType>(type);
            mlir::Type baseType;
            if (!toAny && mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                return castFromUnion(location, type, value, genContext);
            }
        }

        // TODO: issue is with casting to Boolean type from union type for example, you need to cast optional type to boolean to check value
        // get rid of using "OptionalType" and use Union for it with "| undefined"
        // unwrapping optional value to work with union inside, we need it as ' | undefined ' is part of union type
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(valueType))
        {
            if (isa<mlir_ts::UnionType>(optType.getElementType()))
            {
                auto val = V(builder.create<mlir_ts::ValueOrDefaultOp>(location, optType.getElementType(), value));
                CAST_A(unwrappedValue, location, type, val, genContext);            
                return unwrappedValue;
            }
        }        

        // unboxing
        if (auto anyType = dyn_cast<mlir_ts::AnyType>(valueType))
        {
            if (isa<mlir_ts::NumberType>(type) 
                || isa<mlir_ts::BooleanType>(type)
                || isa<mlir_ts::StringType>(type)
                || isa<mlir::IntegerType>(type)
                || isa<mlir::Float32Type>(type)
                || isa<mlir::Float64Type>(type)
                || isa<mlir_ts::ClassType>(type))
            {
                return castFromAny(location, type, value, genContext);
            }
        }

        // opaque to hybrid func
        if (auto opaqueType = dyn_cast<mlir_ts::OpaqueType>(valueType))
        {
            if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                return V(builder.create<mlir_ts::CastOp>(location, type, value));
            }            

            if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
            {
                auto funcValue = builder.create<mlir_ts::CastOp>(
                    location, 
                    mlir_ts::FunctionType::get(builder.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg()), 
                    value);
                return V(builder.create<mlir_ts::CastOp>(location, type, funcValue));
            }
        }

        if (mth.isAnyFunctionType(valueType) && mth.isAnyFunctionType(type)) {

            if (mth.isGenericType(valueType))
            {
                // need to instantiate generic method
                auto result = instantiateSpecializedFunction(location, value, type, genContext);
                EXIT_IF_FAILED(result);
            }

            // fall through to finish cast operation
            if (!mth.CanCastFunctionTypeToFunctionType(valueType, type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();                
            }

            if (!mth.isGenericType(type) && !mth.isGenericType(valueType))
            {
                // test fun types
                auto test = mth.TestFunctionTypesMatchWithObjectMethods(location, valueType, type).result == MatchResultType::Match;
                if (!test)
                {
                    emitError(location) << to_print(valueType) << " is not matching type " << to_print(type);
                    return mlir::failure();
                }
            }
        }

        // cast ext method to bound method
        if (auto extFuncType = dyn_cast<mlir_ts::ExtensionFunctionType>(valueType))
        {
            if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
            {
                auto boundFunc = createBoundMethodFromExtensionMethod(location, value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>());
                return V(builder.create<mlir_ts::CastOp>(location, type, boundFunc));
            }

            if (auto boundFuncType = dyn_cast<mlir_ts::BoundFunctionType>(type))
            {
                auto boundFunc = createBoundMethodFromExtensionMethod(location, value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>());
                return V(builder.create<mlir_ts::CastOp>(location, type, boundFunc));
            }            
        }

        // wrong casts
        // TODO: put it into Cast::Verify
        if (mth.isAnyFunctionType(valueType) && 
            !mth.isAnyFunctionType(type, true) 
            && !isa<mlir_ts::OpaqueType>(type) 
            && !isa<mlir_ts::AnyType>(type)
            && !isa<mlir_ts::BooleanType>(type)) {
            emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
            return mlir::failure();
        }        

        if (isa<mlir_ts::ArrayType>(type) && isa<mlir_ts::TupleType>(valueType) 
            || isa<mlir_ts::TupleType>(type) && isa<mlir_ts::ArrayType>(valueType))
        {
            emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
            return mlir::failure();
        }

        if (auto valueArrayType = dyn_cast<mlir_ts::ArrayType>(valueType))
        {
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
                auto extendsResult = mth.extendsType(location, valueArrayType.getElementType(), arrayType.getElementType(), typeParamsWithArgs);
                if (extendsResult != ExtendsResult::True)
                {
                    emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type) 
                        << " as element type " << to_print(arrayType.getElementType()) << " is not base of type " 
                        << to_print(valueArrayType.getElementType());
                    return mlir::failure();
                }
            }
        }

        return V(builder.create<mlir_ts::CastOp>(location, type, value));
    }

    ValueOrLogicalResult castFromAny(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___as";

        if (!existGenericFunctionMap(funcName))
        {
            // TODO: must be improved, outdated
            auto src = S("function __as<T>(a: any) : T \
                { \
                    if (typeof a == 'number') return a; \
                    if (typeof a == 'string') return a; \
                    if (typeof a == 'i32') return a; \
                    if (typeof a == 'class') if (a instanceof T) return a; \
                    return null; \
                } \
                ");

            {
                MLIRLocationGuard vgLoc(overwriteLoc); 
                overwriteLoc = location;
                if (mlir::failed(parsePartialStatements(src)))
                {
                    assert(false);
                    return mlir::failure();
                }
            }
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        GenContext funcCallGenContext(genContext);
        funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS", type});

        SmallVector<mlir::Value, 4> operands;
        operands.push_back(value);

        NodeFactory nf(NodeFactoryFlags::None);
        return mlirGenCallExpression(location, funcResult, { nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS")).as<Node>()) }, operands, funcCallGenContext);
    }

    ValueOrLogicalResult castFromUnion(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(value.getType()))
        {
            if (auto normalizedUnion = dyn_cast<mlir_ts::UnionType>(mth.getUnionTypeWithMerge(location, unionType.getTypes())))
            {
                // info, we add "_" extra as scanner append "_" in front of "__";
                auto funcName = "___cast";

                // we need to remove current implementation as we have different implementation per union type
                removeGenericFunctionMap(funcName);
                
                // TODO: must be improved
                stringstream ss;

                StringMap<boolean> typeOfs;
                SmallVector<mlir::Type> classInstances;
                ss << S("function __cast<T, U>(t: T) : U {\n");
                for (auto subType : normalizedUnion.getTypes())
                {
                /*
                        if (typeof a == 'number') return a; \
                        if (typeof a == 'string') return a; \
                        if (typeof a == 'i32') return a; \
                        if (typeof a == 'class') if (a instanceof U) return a; \
                        return null; \"
                */
                    mlir::TypeSwitch<mlir::Type>(subType)
                        .Case<mlir_ts::BooleanType>([&](auto _) { typeOfs["boolean"] = true; })
                        .Case<mlir_ts::TypePredicateType>([&](auto _) { typeOfs["boolean"] = true; })
                        .Case<mlir_ts::NumberType>([&](auto _) { typeOfs["number"] = true; })
                        .Case<mlir_ts::StringType>([&](auto _) { typeOfs["string"] = true; })
                        .Case<mlir_ts::CharType>([&](auto _) { typeOfs["char"] = true; })
                        .Case<mlir::IntegerType>([&](auto intType_) {
                            if (intType_.isSignless()) typeOfs["i" + std::to_string(intType_.getWidth())] = true; else
                            if (intType_.isSigned()) typeOfs["s" + std::to_string(intType_.getWidth())] = true; else
                            if (intType_.isUnsigned()) typeOfs["u" + std::to_string(intType_.getWidth())] = true; })
                        .Case<mlir::FloatType>([&](auto floatType_) { typeOfs["f" + std::to_string(floatType_.getWidth())] = true; })
                        .Case<mlir::IndexType>([&](auto _) { typeOfs["index"] = true; })
                        .Case<mlir_ts::HybridFunctionType>([&](auto _) { typeOfs["function"] = true; })
                        .Case<mlir_ts::ClassType>([&](auto classType_) { typeOfs["class"] = true; classInstances.push_back(classType_); })
                        .Case<mlir_ts::InterfaceType>([&](auto _) { typeOfs["interface"] = true; })
                        .Default([&](auto type) { 
                            LLVM_DEBUG(llvm::dbgs() << "\n\t TypeOf NOT IMPLEMENTED for Type: " << type << "\n";);
                            llvm_unreachable("not implemented yet"); 
                        });                                   
                }

                auto next = false;
                for (auto& pair : typeOfs)
                {
                    if (next) ss << S(" else ");

                    ss << S("if (typeof t == '");
                    ss << stows(pair.getKey().str());
                    ss << S("') ");
                    if (pair.getKey() == "class")
                    {
                        ss << S("{ \n");

                        for (auto [index, _] : enumerate(classInstances))
                        {
                            ss << S("if (t instanceof TYPE_INST_ALIAS");
                            ss << index;
                            ss << S(") return t;\n");
                        }

                        ss << S(" }\n");
                    }
                    else
                    {
                        ss << S("return t;\n");
                    }

                    next = true;
                }

                ss << "\nthrow \"Can't cast from union type\";\n";                    
                ss << S("}\n");

                auto src = ss.str();

                {
                    MLIRLocationGuard vgLoc(overwriteLoc); 
                    overwriteLoc = location;

                    if (mlir::failed(parsePartialStatements(src)))
                    {
                        assert(false);
                        return mlir::failure();
                    }
                }

                auto funcResult = resolveIdentifier(location, funcName, genContext);

                assert(funcResult);

                GenContext funcCallGenContext(genContext);
                funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", value.getType()});
                funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_U", type});

                for (auto [index, instanceOfType] : enumerate(classInstances))
                {
                    funcCallGenContext.typeAliasMap.insert({"TYPE_INST_ALIAS" + std::to_string(index), instanceOfType});
                }

                SmallVector<mlir::Value, 4> operands;
                operands.push_back(value);

                NodeFactory nf(NodeFactoryFlags::None);
                return mlirGenCallExpression(
                    location, 
                    funcResult, 
                    { 
                        nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_T")).as<Node>()), 
                        nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_U")).as<Node>()) 
                    }, 
                    operands, 
                    funcCallGenContext);
            }
        }

        return mlir::failure();
    }    

    ValueOrLogicalResult castTupleToInterface(mlir::Location location, mlir::Value in, mlir::Type tupleTypeIn,
                                     mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {

        auto tupleType = mth.convertConstTupleTypeToTupleType(tupleTypeIn);
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());

        auto inEffective = in;

        auto srcTuple = mlir::cast<mlir_ts::TupleType>(tupleType);
        if (mlir::failed(mth.canCastTupleToInterface(location, srcTuple, interfaceInfo, true)))
        {
            SmallVector<mlir_ts::FieldInfo> fields;
            if (mlir::failed(interfaceInfo->getTupleTypeFields(fields, builder.getContext())))
            {
                return mlir::failure();
            }

            // append all fields from original tuple
            for (auto origField : srcTuple.getFields()) {
                if (std::find_if(
                    fields.begin(), 
                    fields.end(), 
                    [&] (auto& item) { 
                        return item.id == origField.id; 
                    }) == fields.end())
                {
                    fields.push_back(origField);
                }                
            }

            auto newInterfaceTupleType = getTupleType(fields);
            CAST(inEffective, location, newInterfaceTupleType, inEffective, genContext);
            tupleType = newInterfaceTupleType;

            emitWarning(location, "") << "Cloned object is used. Ensure all types are matching to interface: " << interfaceInfo->fullName;
        }

        // TODO: finish it, what to finish it? maybe optimization not to create extra object?
        // convert Tuple to Object
        auto objType = mlir_ts::ObjectType::get(tupleType);
        auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(tupleType), builder.getBoolAttr(false));
        builder.create<mlir_ts::StoreOp>(location, inEffective, valueAddr);
        auto inCasted = builder.create<mlir_ts::CastOp>(location, objType, valueAddr);

        return castObjectToInterface(location, inCasted, objType, interfaceInfo, genContext);
    }

    ValueOrLogicalResult castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        return castObjectToInterface(location, in, objType, interfaceInfo, genContext);
    }

    ValueOrLogicalResult castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    InterfaceInfo::TypePtr interfaceInfo, const GenContext &genContext)
    {
        auto result = mlirGenCreateInterfaceVTableForObject(location, in, objType, interfaceInfo, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto createdInterfaceVTableForObject = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                << "@ created interface:" << createdInterfaceVTableForObject << "\n";);

        return V(builder.create<mlir_ts::NewInterfaceOp>(location, 
            mlir::TypeRange{interfaceInfo->interfaceType}, in, createdInterfaceVTableForObject));
    }    

    mlir_ts::CreateBoundFunctionOp createBoundMethodFromExtensionMethod(mlir::Location location, mlir_ts::CreateExtensionFunctionOp createExtentionFunction)
    {
        auto extFuncType = createExtentionFunction.getType();
        auto boundFuncVal = builder.create<mlir_ts::CreateBoundFunctionOp>(
            location, 
            getBoundFunctionType(
                extFuncType.getInputs(), 
                extFuncType.getResults(), 
                extFuncType.isVarArg()), 
            createExtentionFunction.getThisVal(), createExtentionFunction.getFunc());            

        return boundFuncVal;
    }

    mlir::Type getType(Node typeReferenceAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)typeReferenceAST;
        if (kind == SyntaxKind::BooleanKeyword)
        {
            return getBooleanType();
        }
        else if (kind == SyntaxKind::NumberKeyword)
        {
            return getNumberType();
        }
        else if (kind == SyntaxKind::BigIntKeyword)
        {
            return getBigIntType();
        }
        else if (kind == SyntaxKind::StringKeyword)
        {
            return getStringType();
        }
        else if (kind == SyntaxKind::VoidKeyword)
        {
            return getVoidType();
        }
        else if (kind == SyntaxKind::FunctionType)
        {
            return getFunctionType(typeReferenceAST.as<FunctionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructorType)
        {
            // TODO: do I need to add flag to FunctionType to show that this is ConstructorType?
            return getConstructorType(typeReferenceAST.as<ConstructorTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::CallSignature)
        {
            return getCallSignature(typeReferenceAST.as<CallSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::MethodSignature)
        {
            return getMethodSignature(typeReferenceAST.as<MethodSignature>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructSignature)
        {
            return getConstructSignature(typeReferenceAST.as<ConstructSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            return getIndexSignature(typeReferenceAST.as<IndexSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::TupleType)
        {
            return getTupleType(typeReferenceAST.as<TupleTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeLiteral)
        {
            // TODO: review it, I think it should be ObjectType
            // return getObjectType(getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext));
            return getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayType)
        {
            return getArrayType(typeReferenceAST.as<ArrayTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::UnionType)
        {
            return getUnionType(typeReferenceAST.as<UnionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::IntersectionType)
        {
            return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedType)
        {
            return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::LiteralType)
        {
            return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
        }
        else if (kind == SyntaxKind::TypeReference)
        {
            return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeQuery)
        {
            return getTypeByTypeQuery(typeReferenceAST.as<TypeQueryNode>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectKeyword)
        {
            return getObjectType(getAnyType());
        }
        else if (kind == SyntaxKind::AnyKeyword)
        {
            return getAnyType();
        }
        else if (kind == SyntaxKind::UnknownKeyword)
        {
            // TODO: do I need to have special type?
            return getUnknownType();
        }
        else if (kind == SyntaxKind::SymbolKeyword)
        {
            return getSymbolType();
        }
        else if (kind == SyntaxKind::UndefinedKeyword)
        {
            return getUndefinedType();
        }
        else if (kind == SyntaxKind::TypePredicate)
        {
            // in runtime it is boolean (it is needed to track types)
            return getTypePredicateType(typeReferenceAST.as<TypePredicateNode>(), genContext);
        }
        else if (kind == SyntaxKind::ThisType)
        {
            if (genContext.thisType)
            {
                return genContext.thisType;
            }
            
            NodeFactory nf(NodeFactoryFlags::None);
            auto thisType = evaluate(nf.createToken(SyntaxKind::ThisKeyword), genContext);
            LLVM_DEBUG(llvm::dbgs() << "\n!! this type from variable: [" << thisType << "]\n";);
            return thisType;
        }
        else if (kind == SyntaxKind::Unknown)
        {
            return getUnknownType();
        }
        else if (kind == SyntaxKind::ConditionalType)
        {
            return getConditionalType(typeReferenceAST.as<ConditionalTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOperator)
        {
            return getTypeOperator(typeReferenceAST.as<TypeOperatorNode>(), genContext);
        }
        else if (kind == SyntaxKind::IndexedAccessType)
        {
            return getIndexedAccessType(typeReferenceAST.as<IndexedAccessTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::MappedType)
        {
            return getMappedType(typeReferenceAST.as<MappedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateLiteralType)
        {
            return getTemplateLiteralType(typeReferenceAST.as<TemplateLiteralTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeParameter)
        {
            return getResolveTypeParameter(typeReferenceAST.as<TypeParameterDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::InferType)
        {
            return getInferType(loc(typeReferenceAST), typeReferenceAST.as<InferTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::OptionalType)
        {
            return getOptionalType(typeReferenceAST.as<OptionalTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::RestType)
        {
            return getRestType(typeReferenceAST.as<RestTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::NeverKeyword)
        {
            return getNeverType();
        }

        llvm_unreachable("not implemented type declaration");
        // return getAnyType();
    }

    mlir::Type getInferType(mlir::Location location, InferTypeNode inferTypeNodeAST, const GenContext &genContext)
    {
        auto type = getType(inferTypeNodeAST->typeParameter, genContext);
        auto inferType = getInferType(type);

        LLVM_DEBUG(llvm::dbgs() << "\n!! infer type [" << inferType << "]\n";);

        // TODO: review function 'extends' in MLIRTypeHelper with the same logic adding infer types to context
        auto &typeParamsWithArgs = const_cast<GenContext &>(genContext).typeParamsWithArgs;
        mth.appendInferTypeToContext(location, type, inferType, typeParamsWithArgs);

        return inferType;
    }

    mlir::Type getResolveTypeParameter(StringRef typeParamName, bool defaultType, const GenContext &genContext)
    {
        // to build generic type with generic names
        auto foundAlias = genContext.typeAliasMap.find(typeParamName);
        if (foundAlias != genContext.typeAliasMap.end())
        {
            auto type = (*foundAlias).getValue();
            // LLVM_DEBUG(llvm::dbgs() << "\n!! type gen. param as alias [" << typeParamName << "] -> [" << type
            //                         << "]\n";);
            return type;
        }

        auto found = genContext.typeParamsWithArgs.find(typeParamName);
        if (found != genContext.typeParamsWithArgs.end())
        {
            auto type = (*found).getValue().second;
            //LLVM_DEBUG(llvm::dbgs() << "\n!! type gen. param [" << typeParamName << "] -> [" << type << "]\n";);
            return type;
        }

        if (defaultType)
        {
            // unresolved generic
            return getNamedGenericType(typeParamName);
        }

        // name is not found
        return mlir::Type();
    }

    mlir::Type getResolveTypeParameter(TypeParameterDeclaration typeParameterDeclaration, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(typeParameterDeclaration->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::Type();
        }

        return getResolveTypeParameter(name, true, genContext);
    }

    mlir::Type getTypeByTypeName(Node node, const GenContext &genContext)
    {
        if (node == SyntaxKind::Identifier)
        {
            auto name = MLIRHelper::getName(node);
            return resolveTypeByName(loc(node), name, genContext);
        }        
        else if (node == SyntaxKind::QualifiedName)
        {
            // TODO: it seems namespace access, can u optimize it somehow?
            auto result = mlirGen(node.as<QualifiedName>(), genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Type();
            }

            auto val = V(result);
            return val.getType();
        }
        
        llvm_unreachable("not implemented");
    }

    mlir::Type getFirstTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        return getType(typeArguments->front(), genContext);
    }

    mlir::Type getSecondTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        return getType(typeArguments[1], genContext);
    }

    Reason testConstraint(mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, const GenContext &genContext) {
        // we need to add current type into context to be able to use it in resolving "extends" constraints
        GenContext constraintGenContext(genContext);
        for (auto &typeParamWithArg : pairs)
        {
            constraintGenContext.typeParamsWithArgs.insert({typeParamWithArg.getKey(), typeParamWithArg.getValue()});
        }

        constraintGenContext.typeParamsWithArgs.insert({typeParam->getName(), std::make_pair(typeParam, type)});

        auto constraintType = getType(typeParam->getConstraint(), constraintGenContext);
        if (!constraintType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed. should be resolved later\n";);
            return Reason::Failure;
        }

        auto extendsResult = mth.extendsType(location, type, constraintType, pairs);
        if (extendsResult != ExtendsResult::True)
        {
            // special case when we work with generic type(which are not specialized yet)
            if (mth.isGenericType(type))
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, type)});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: " << type << " (because of generic).";);
                return Reason::None;                    
            }

            if (extendsResult == ExtendsResult::Any)
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, getAnyType())});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: any.";);
                return Reason::None;                    
            }                

            if (extendsResult == ExtendsResult::Never)
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, getNeverType())});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: never.";);
                return Reason::None;                    
            }

            LLVM_DEBUG(llvm::dbgs() << "Type " << type << " does extend "
                                    << constraintType << ".";);

            emitWarning(location, "") << "Type " << type << " does not satisfy the constraint "
                                    << constraintType << ".";

            return Reason::FailedConstraint;
        }

        return Reason::NoConstraint;
    }

    std::tuple<mlir::LogicalResult, IsGeneric> zipTypeParameterWithArgument(
        mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, bool noExtendTest,
        const GenContext &genContext, bool mergeTypes = false, bool arrayMerge = false)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! assigning generic type: " << typeParam->getName() << " type: " << type
                                << "\n";);

        if (mth.isNoneType(type))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed.\n";);
            return {mlir::failure(), IsGeneric::False};
        }

        if (isa<mlir_ts::NamedGenericType>(type))
        {
            pairs.insert({typeParam->getName(), std::make_pair(typeParam, type)});
            return {mlir::success(), IsGeneric::True};
        }

        auto name = typeParam->getName();
        auto existType = pairs.lookup(name);
        if (existType.second)
        {
            if (existType.second != type)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! replacing existing type for: " << name
                                        << " exist type: " << existType.second << " new type: " << type << "\n";);

                if (!isa<mlir_ts::NamedGenericType>(existType.second) && mergeTypes)
                {
                    auto merged = false;
                    if (arrayMerge)
                    {
                        type = mth.arrayMergeType(location, existType.second, type, merged);
                    }
                    else
                    {
                        type = mth.mergeType(location, existType.second, type, merged);
                    }

                    LLVM_DEBUG(llvm::dbgs() << "\n!! result (after merge) type: " << type << "\n";);
                }

                // TODO: Do I need to join types?
                pairs[name] = std::make_pair(typeParam, type);
            }
        }
        else
        {
            pairs.insert({name, std::make_pair(typeParam, type)});
        }

        return {mlir::success(), IsGeneric::False};
    }

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, llvm::ArrayRef<mlir::Type> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            auto type = index < argsCount
                            ? typeArgs[index]
                            : (isDefault = true, typeParam->hasDefault() 
                                ? getType(typeParam->getDefault(), genContext) 
                                : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type());
            if (!type)
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }


    std::tuple<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            auto type = index < argsCount
                            ? getType(typeArgs[index], genContext)
                            : (isDefault = true, typeParam->hasDefault() 
                                ? getType(typeParam->getDefault(), genContext) 
                                : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type());
            if (!type)
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArgumentsNoDefaults(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            auto type = index < argsCount
                            ? getType(typeArgs[index], genContext)
                            : (isDefault = true,
                               typeParam->hasDefault() 
                               ? getType(typeParam->getDefault(), genContext) 
                               : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type());
            if (!type)
            {
                return {mlir::success(), anyNamedGenericType};
            }

            if (isDefault)
            {
                return {mlir::success(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithDefaultArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs ? typeArgs.size() : 0;
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            if (index < argsCount)
            {
                // we need to process only default values
                continue;
            }
            auto type = typeParam->hasDefault() 
                            ? getType(typeParam->getDefault(), genContext) 
                            : typeParam->hasConstraint() 
                                ? getType(typeParam->getConstraint(), genContext) 
                                : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type();
            if (!type)
            {
                return {mlir::success(), anyNamedGenericType};
            }

            auto name = typeParam->getName();
            auto existType = pairs.lookup(name);
            if (existType.second)
            {
                // type is resolved
                continue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! adding default type: " << typeParam->getName() << " type: " << type
                                << "\n";);

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    mlir::Type createTypeReferenceType(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        mlir::SmallVector<mlir::Type> typeArgs;
        for (auto typeArgNode : typeReferenceAST->typeArguments)
        {
            auto typeArg = getType(typeArgNode, genContext);
            if (!typeArg)
            {
                return mlir::Type();
            }

            typeArgs.push_back(typeArg);
        }

        auto nameRef = MLIRHelper::getName(typeReferenceAST->typeName, stringAllocator);
        auto typeRefType = getTypeReferenceType(nameRef, typeArgs);

        LLVM_DEBUG(llvm::dbgs() << "\n!! generic TypeReferenceType: " << typeRefType;);

        return typeRefType;
    };

    mlir::Type getTypeByTypeReference(mlir::Location location, mlir_ts::TypeReferenceType typeReferenceType, const GenContext &genContext)
    {
        // check utility types
        auto name = typeReferenceType.getName().getValue();

        // try to resolve from type alias first
        auto genericTypeAliasInfo = lookupGenericTypeAliasMap(name);
        if (!is_default(genericTypeAliasInfo))
        {
            GenContext genericTypeGenContext(genContext);

            auto typeParams = std::get<0>(genericTypeAliasInfo);
            auto typeNode = std::get<1>(genericTypeAliasInfo);

            auto [result, hasAnyNamedGenericType] =
                zipTypeParametersWithArguments(location, typeParams, typeReferenceType.getTypes(),
                                               genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

            if (mlir::failed(result))
            {
                return mlir::Type();
            }

            return getType(typeNode, genericTypeGenContext);
        }  

        return mlir::Type();      
    }

    mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        auto location = loc(typeReferenceAST);

        // check utility types
        auto name = MLIRHelper::getName(typeReferenceAST->typeName);

        if (typeReferenceAST->typeArguments.size())
        {
            // try to resolve from type alias first
            auto genericTypeAliasInfo = lookupGenericTypeAliasMap(name);
            if (!is_default(genericTypeAliasInfo))
            {
                GenContext genericTypeGenContext(genContext);

                auto typeParams = std::get<0>(genericTypeAliasInfo);
                auto typeNode = std::get<1>(genericTypeAliasInfo);

                auto [result, hasAnyNamedGenericType] =
                    zipTypeParametersWithArguments(location, typeParams, typeReferenceAST->typeArguments,
                                                genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

                if (mlir::failed(result))
                {
                    return mlir::Type();
                }

                if (hasAnyNamedGenericType == IsGeneric::True)
                {
                    return createTypeReferenceType(typeReferenceAST, genericTypeGenContext);
                }

                return getType(typeNode, genericTypeGenContext);
            }

            if (auto genericClassTypeInfo = lookupGenericClassesMap(name))
            {
                auto classType = genericClassTypeInfo->classType;
                auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                                                        typeReferenceAST->typeArguments, genContext, true);
                if (mlir::succeeded(result))
                {
                    return specType;
                }

                return classType;
            }

            if (auto genericInterfaceTypeInfo = lookupGenericInterfacesMap(name))
            {
                auto interfaceType = genericInterfaceTypeInfo->interfaceType;
                auto [result, specType] = instantiateSpecializedInterfaceType(location, interfaceType,
                                                                            typeReferenceAST->typeArguments, genContext, true);
                if (mlir::succeeded(result))
                {
                    return specType;
                }

                return interfaceType;
            }

            if (auto embedType = findEmbeddedType(location, name, typeReferenceAST->typeArguments, genContext))
            {
                return embedType;
            }
        }

        if (auto type = getTypeByTypeName(typeReferenceAST->typeName, genContext))
        {
            return type;
        }

        if (auto embedType = findEmbeddedType(location, name, typeReferenceAST->typeArguments, genContext))
        {
            return embedType;
        }

        return mlir::Type();
    }

    mlir::Type findEmbeddedType(mlir::Location location, std::string name, NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        auto typeArgumentsSize = typeArguments->size();
        if (typeArgumentsSize == 0)
        {
            if (auto type = getEmbeddedType(name))
            {
                return type;
            }
        }

        if (typeArgumentsSize == 1)
        {
            if (auto type = getEmbeddedTypeWithParam(name, typeArguments, genContext))
            {
                return type;
            }
        }

        if (typeArgumentsSize > 1)
        {
            if (auto type = getEmbeddedTypeWithManyParams(location, name, typeArguments, genContext))
            {
                return type;
            }
        }

        return mlir::Type();
    }

    bool isEmbededType(mlir::StringRef name)
    {
        return compileOptions.enableBuiltins ? isEmbededTypeWithBuiltins(name) : isEmbededTypeWithNoBuiltins(name);
    }
    
    bool isEmbededTypeWithBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<bool> embeddedTypes {
            {"TemplateStringsArray", true },
            {"const", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", true },
            {"Object", true },
            {"String", true },
            {"Boolean", true },
            {"Function", true },
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", true },
            {"short", true },
            {"ushort", true },
            {"int", true },
            {"uint", true },
            {"index", true },
            {"long", true },
            {"ulong", true },
            {"char", true },
            {"i8", true },
            {"i16", true },
            {"i32", true },
            {"i64", true },
            {"u8", true},
            {"u16", true},
            {"u32", true},
            {"u64", true},
            {"s8", true},
            {"s16", true},
            {"s32", true},
            {"s64", true},
            {"f16", true},
            {"f32", true},
            {"f64", true},
            {"f128", true},
            {"half", true},
            {"float", true},
            {"double", true},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS
            {"Int8Array", true },
            {"Uint8Array", true },
            {"Int16Array", true },
            {"Uint16Array", true },
            {"Int32Array", true },
            {"Uint32Array", true },
            {"BigInt64Array", true },
            {"BigUint64Array", true },
            {"Float16Array", true },
            {"Float32Array", true },
            {"Float64Array", true },
            {"Float128Array", true},
#endif

            {"TypeOf", true },
            {"Opaque", true }, // to support void*
            {"Reference", true }, // to support dll import
            {"Readonly", true },
            {"Partial", true },
            {"Required", true },
            {"ThisType", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Awaited", true },
            {"Promise", true },
#endif            
            {"NonNullable", true },
            {"Array", true },
            {"ReadonlyArray", true },
            {"ReturnType", true },
            {"Parameters", true },
            {"ConstructorParameters", true },
            {"ThisParameterType", true },
            {"OmitThisParameter", true },
            {"Uppercase", true },
            {"Lowercase", true },
            {"Capitalize", true },
            {"Uncapitalize", true },
            {"Exclude",  true },
            {"Extract", true },
            {"Pick", true },
            {"Omit",  true },
            {"Record", true },
        };

        auto type = embeddedTypes[name];
        return type;
    }

    bool isEmbededTypeWithNoBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<bool> embeddedTypes {
            {"TemplateStringsArray", true },
            {"const", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", true },
            {"Object", true },
            {"String", true },
            {"Boolean", true },
            {"Function", true },
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", true },
            {"short", true },
            {"ushort", true },
            {"int", true },
            {"uint", true },
            {"index", true },
            {"long", true },
            {"ulong", true },
            {"char", true },
            {"i8", true },
            {"i16", true },
            {"i32", true },
            {"i64", true },
            {"u8", true},
            {"u16", true},
            {"u32", true},
            {"u64", true},
            {"s8", true},
            {"s16", true},
            {"s32", true},
            {"s64", true},
            {"f16", true},
            {"f32", true},
            {"f64", true},
            {"f128", true},
            {"half", true},
            {"float", true},
            {"double", true},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS_NOBUILTINS
            {"Int8Array", true },
            {"Uint8Array", true },
            {"Int16Array", true },
            {"Uint16Array", true },
            {"Int32Array", true },
            {"Uint32Array", true },
            {"BigInt64Array", true },
            {"BigUint64Array", true },
            {"Float16Array", true },
            {"Float32Array", true },
            {"Float64Array", true },
            {"Float128Array", true},
#endif

            {"TypeOf", true },
            {"Opaque", true }, // to support void*
            {"Reference", true }, // to support dll import
            {"ThisType", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Awaited", true },
            {"Promise", true },
#endif            
            {"Array", true }
        };

        auto type = embeddedTypes[name];
        return type;
    }

    mlir::Type getEmbeddedType(mlir::StringRef name)
    {
        return compileOptions.enableBuiltins ? getEmbeddedTypeBuiltins(name) : getEmbeddedTypeNoBuiltins(name);
    }

    mlir::Type getEmbeddedTypeBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<mlir::Type> embeddedTypes {
            {"TemplateStringsArray", getArrayType(getStringType()) },
            {"const",getConstType() },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", getNumberType() },
            {"Object", getObjectType(getAnyType()) },
            {"String", getStringType()},
            {"Boolean", getBooleanType()},
            {"Function", getFunctionType({getArrayType(getAnyType())}, {getAnyType()}, true)},
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", builder.getIntegerType(8) },
            {"short", builder.getIntegerType(16, true) },
            {"ushort", builder.getIntegerType(16, false) },
            {"int", builder.getIntegerType(32, true) },
            {"uint", builder.getIntegerType(32, false) },
            {"index", builder.getIndexType() },
            {"long", builder.getIntegerType(64, true) },
            {"ulong", builder.getIntegerType(64, false) },
            {"char", getCharType() },
            {"i8", builder.getIntegerType(8) },
            {"i16", builder.getIntegerType(16) },
            {"i32", builder.getIntegerType(32) },
            {"i64", builder.getIntegerType(64) },
            {"u8", builder.getIntegerType(8, false)},
            {"u16", builder.getIntegerType(16, false)},
            {"u32", builder.getIntegerType(32, false)},
            {"u64", builder.getIntegerType(64, false)},
            {"s8", builder.getIntegerType(8, true) },
            {"s16", builder.getIntegerType(16, true) },
            {"s32", builder.getIntegerType(32, true) },
            {"s64", builder.getIntegerType(64, true) },
            {"f16", builder.getF16Type()},
            {"f32", builder.getF32Type()},
            {"f64", builder.getF64Type()},
            {"f128", builder.getF128Type()},
            {"half", builder.getF16Type()},
            {"float", builder.getF32Type()},
            {"double", builder.getF64Type()},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS
            {"Int8Array", getArrayType(builder.getIntegerType(8, true)) },
            {"Uint8Array", getArrayType(builder.getIntegerType(8, false))},
            {"Int16Array", getArrayType(builder.getIntegerType(16, true)) },
            {"Uint16Array", getArrayType(builder.getIntegerType(16, false))},
            {"Int32Array", getArrayType(builder.getIntegerType(32, true)) },
            {"Uint32Array", getArrayType(builder.getIntegerType(32, false))},
            {"BigInt64Array", getArrayType(builder.getIntegerType(64, true)) },
            {"BigUint64Array", getArrayType(builder.getIntegerType(64, false))},
            {"Float16Array", getArrayType(builder.getF16Type())},
            {"Float32Array", getArrayType(builder.getF32Type())},
            {"Float64Array", getArrayType(builder.getF64Type())},
            {"Float128Array", getArrayType(builder.getF128Type())},
#endif
            {"Opaque", getOpaqueType()},
        };

        auto type = embeddedTypes[name];
        return type;
    }

    mlir::Type getEmbeddedTypeNoBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<mlir::Type> embeddedTypes {
            {"TemplateStringsArray", getArrayType(getStringType()) },
            {"const",getConstType() },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", getNumberType() },
            {"Object", getObjectType(getAnyType()) },
            {"String", getStringType()},
            {"Boolean", getBooleanType()},
            {"Function", getFunctionType({getArrayType(getAnyType())}, {getAnyType()}, true)},
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", builder.getIntegerType(8) },
            {"short", builder.getIntegerType(16, true) },
            {"ushort", builder.getIntegerType(16, false) },
            {"int", builder.getIntegerType(32, true) },
            {"uint", builder.getIntegerType(32, false) },
            {"index", builder.getIndexType() },
            {"long", builder.getIntegerType(64, true) },
            {"ulong", builder.getIntegerType(64, false) },
            {"char", getCharType() },
            {"i8", builder.getIntegerType(8) },
            {"i16", builder.getIntegerType(16) },
            {"i32", builder.getIntegerType(32) },
            {"i64", builder.getIntegerType(64) },
            {"u8", builder.getIntegerType(8, false)},
            {"u16", builder.getIntegerType(16, false)},
            {"u32", builder.getIntegerType(32, false)},
            {"u64", builder.getIntegerType(64, false)},
            {"s8", builder.getIntegerType(8, true) },
            {"s16", builder.getIntegerType(16, true) },
            {"s32", builder.getIntegerType(32, true) },
            {"s64", builder.getIntegerType(64, true) },
            {"f16", builder.getF16Type()},
            {"f32", builder.getF32Type()},
            {"f64", builder.getF64Type()},
            {"f128", builder.getF128Type()},
            {"half", builder.getF16Type()},
            {"float", builder.getF32Type()},
            {"double", builder.getF64Type()},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS_NOBUILTINS
            {"Int8Array", getArrayType(builder.getIntegerType(8, true)) },
            {"Uint8Array", getArrayType(builder.getIntegerType(8, false))},
            {"Int16Array", getArrayType(builder.getIntegerType(16, true)) },
            {"Uint16Array", getArrayType(builder.getIntegerType(16, false))},
            {"Int32Array", getArrayType(builder.getIntegerType(32, true)) },
            {"Uint32Array", getArrayType(builder.getIntegerType(32, false))},
            {"BigInt64Array", getArrayType(builder.getIntegerType(64, true)) },
            {"BigUint64Array", getArrayType(builder.getIntegerType(64, false))},
            {"Float16Array", getArrayType(builder.getF16Type())},
            {"Float32Array", getArrayType(builder.getF32Type())},
            {"Float64Array", getArrayType(builder.getF64Type())},
            {"Float128Array", getArrayType(builder.getF128Type())},
#endif

            {"Opaque", getOpaqueType()},
        };

        auto type = embeddedTypes[name];
        return type;
    }    

    mlir::Type getEmbeddedTypeWithParam(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        return compileOptions.enableBuiltins 
            ? getEmbeddedTypeWithParamBuiltins(name, typeArguments, genContext) 
            : getEmbeddedTypeWithParamNoBuiltins(name, typeArguments, genContext);
    }

    mlir::Type getEmbeddedTypeWithParamBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        auto translate = llvm::StringSwitch<std::function<mlir::Type(NodeArray<TypeNode> &, const GenContext &)>>(name)
            .Case("TypeOf", [&] (auto typeArguments, auto genContext) {
                auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
                type = mth.wideStorageType(type);
                return type;
            })
            .Case("Reference", [&] (auto typeArguments, auto genContext) {
                auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return mlir_ts::RefType::get(type);
            })
            .Case("Readonly", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
            .Case("Partial", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
            .Case("Required", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
            .Case("ThisType", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
#ifdef ENABLE_JS_BUILTIN_TYPES
            .Case("Awaited", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
            .Case("Promise", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
#endif            
            .Case("NonNullable", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return NonNullableTypes(elemnentType);
            })
            .Case("Array", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return getArrayType(elemnentType);
            })
            .Case("ReadonlyArray", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return getArrayType(elemnentType);
            })
            .Case("ReturnType", [&] (auto typeArguments, auto genContext) {
                auto elementType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ReturnType Of: " << elementType;);
                auto retType = mth.getReturnTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            })
            .Case("Parameters", [&] (auto typeArguments, auto genContext) {
                auto elementType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = mth.getParamsTupleTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            })
            .Case("ConstructorParameters", [&] (auto typeArguments, auto genContext) {
                auto elementType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = mth.getParamsTupleTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            })
            .Case("ThisParameterType", [&] (auto typeArguments, auto genContext) {
                auto elementType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = mth.getFirstParamFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            })
            .Case("OmitThisParameter", [&] (auto typeArguments, auto genContext) {
                auto elementType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = mth.getOmitThisFunctionTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            })
            .Case("Uppercase", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return UppercaseType(elemnentType);
            })
            .Case("Lowercase", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return LowercaseType(elemnentType);
            })
            .Case("Capitalize", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return CapitalizeType(elemnentType);
            })
            .Case("Uncapitalize", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return UncapitalizeType(elemnentType);
            })
            .Default([] (auto, auto) {
                return mlir::Type();
            });

        return translate(typeArguments, genContext);
    }

    mlir::Type getEmbeddedTypeWithParamNoBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        auto translate = llvm::StringSwitch<std::function<mlir::Type(NodeArray<TypeNode> &, const GenContext &)>>(name)
            .Case("TypeOf", [&] (auto typeArguments, auto genContext) {
                auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
                type = mth.wideStorageType(type);
                return type;
            })
            .Case("Reference", [&] (auto typeArguments, auto genContext) {
                auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return mlir_ts::RefType::get(type);
            })
            .Case("ThisType", std::bind(&MLIRGenImpl::getFirstTypeFromTypeArguments, this, std::placeholders::_1, std::placeholders::_2))
            .Case("Array", [&] (auto typeArguments, auto genContext) {
                auto elemnentType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                return getArrayType(elemnentType);
            })
            .Default([] (auto, auto) {
                return mlir::Type();
            });

        return translate(typeArguments, genContext);
    }

    mlir::Type getEmbeddedTypeWithManyParams(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext)
    {
        return compileOptions.enableBuiltins 
            ? getEmbeddedTypeWithManyParamsBuiltins(location, name, typeArguments, genContext) 
            : mlir::Type();
    }

    mlir::Type getEmbeddedTypeWithManyParamsBuiltins(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext)
    {
        auto translate = llvm::StringSwitch<std::function<mlir::Type(NodeArray<TypeNode> &, const GenContext &)>>(name)
            .Case("Exclude", [&] (auto typeArguments, auto genContext) {
                auto firstType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                auto secondType = getSecondTypeFromTypeArguments(typeArguments, genContext);
                return ExcludeTypes(location, firstType, secondType);
            })
            .Case("Extract", [&] (auto typeArguments, auto genContext) {
                auto firstType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                auto secondType = getSecondTypeFromTypeArguments(typeArguments, genContext);
                return ExtractTypes(location, firstType, secondType);
            })
            .Case("Pick", [&] (auto typeArguments, auto genContext) {
                auto sourceType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                auto keysType = getSecondTypeFromTypeArguments(typeArguments, genContext);
                return PickTypes(sourceType, keysType);
            })
            .Case("Omit", [&] (auto typeArguments, auto genContext) {
                auto sourceType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                auto keysType = getSecondTypeFromTypeArguments(typeArguments, genContext);
                return OmitTypes(sourceType, keysType);
            })
            .Case("Record", [&] (auto typeArguments, auto genContext) {
                auto keysType = getFirstTypeFromTypeArguments(typeArguments, genContext);
                auto sourceType = getSecondTypeFromTypeArguments(typeArguments, genContext);
                return RecordType(keysType, sourceType);
            })
            .Default([] (auto, auto) {
                return mlir::Type();
            });

        return translate(typeArguments, genContext);
    }

    mlir::Type StringLiteralTypeFunc(mlir::Type type, std::function<std::string(StringRef)> f)
    {
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            if (isa<mlir_ts::StringType>(literalType.getElementType()))
            {
                auto newStr = f(mlir::cast<mlir::StringAttr>(literalType.getValue()).getValue());
                auto copyVal = StringRef(newStr).copy(stringAllocator);
                return mlir_ts::LiteralType::get(builder.getStringAttr(copyVal), getStringType());
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't apply string literal type for:" << type << "\n";);

        return mlir::Type();
    }

    mlir::Type UppercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.upper(); });
    }

    mlir::Type LowercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.lower(); });
    }

    mlir::Type CapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type,
                                     [](auto val) { return val.slice(0, 1).upper().append(val.slice(1, val.size())); });
    }

    mlir::Type UncapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type,
                                     [](auto val) { return val.slice(0, 1).lower().append(val.slice(1, val.size())); });
    }

    mlir::Type NonNullableTypes(mlir::Type type)
    {
        if (mth.isGenericType(type))
        {
            return type;
        }

        SmallPtrSet<mlir::Type, 2> types;

        MLIRHelper::flatUnionTypes(types, type);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            if (isa<mlir_ts::NullType>(item) || item == getUndefinedType())
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    // TODO: remove using those types as there issue with generic types
    mlir::Type ExcludeTypes(mlir::Location location, mlir::Type type, mlir::Type exclude)
    {
        if (mth.isGenericType(type) || mth.isGenericType(exclude))
        {
            return getAnyType();
        }

        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> excludeTypes;

        MLIRHelper::flatUnionTypes(types, type);
        MLIRHelper::flatUnionTypes(excludeTypes, exclude);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            // TODO: should I use TypeParamsWithArgs from genContext?
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> emptyTypeParamsWithArgs;
            if (llvm::any_of(excludeTypes, [&](mlir::Type type) { 
                return isTrue(mth.extendsType(location, item, type, emptyTypeParamsWithArgs)); 
            }))
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    mlir::Type ExtractTypes(mlir::Location location, mlir::Type type, mlir::Type extract)
    {
        if (mth.isGenericType(type) || mth.isGenericType(extract))
        {
            return getAnyType();
        }

        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> extractTypes;

        MLIRHelper::flatUnionTypes(types, type);
        MLIRHelper::flatUnionTypes(extractTypes, extract);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            // TODO: should I use TypeParamsWithArgs from genContext?
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> emptyTypeParamsWithArgs;
            if (llvm::any_of(extractTypes, [&](mlir::Type type) { 
                return isTrue(mth.extendsType(location, item, type, emptyTypeParamsWithArgs)); 
            }))
            {
                resTypes.push_back(item);
            }
        }

        return getUnionType(resTypes);
    }

    mlir::Type RecordType(mlir::Type keys, mlir::Type valueType)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Record: " << valueType << ", keys: " << keys << "\n";);
        
        SmallVector<mlir_ts::FieldInfo> fields;

        auto addTypeProcessKey = [&](mlir::Type keyType)
        {
            // get string
            if (auto litType = dyn_cast<mlir_ts::LiteralType>(keyType))
            {
                fields.push_back({ litType.getValue(), valueType });
            }
        };

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
        {
            for (auto keyType : unionType.getTypes())
            {
                addTypeProcessKey(keyType);
            }
        }
        else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
        {
            addTypeProcessKey(litType);
        }
        else
        {
            llvm_unreachable("not implemented");
        }        

        return getTupleType(fields);
    }

    mlir::Type PickTypes(mlir::Type type, mlir::Type keys)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Pick: " << type << ", keys: " << keys << "\n";);

        if (!keys)
        {
            return mlir::Type();
        }

        if (mth.isGenericType(type))
        {
            return getAnyType();
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> pickedTypes;
            for (auto subType : unionType)
            {
                pickedTypes.push_back(PickTypes(subType, keys));
            }

            return getUnionType(pickedTypes);
        }

        SmallVector<mlir_ts::FieldInfo> pickedFields;
        SmallVector<mlir_ts::FieldInfo> fields;
        if (mlir::succeeded(mth.getFields(type, fields)))
        {
            auto pickTypesProcessKey = [&](mlir::Type keyType)
            {
                // get string
                if (auto litType = dyn_cast<mlir_ts::LiteralType>(keyType))
                {
                    // find field
                    auto found = std::find_if(fields.begin(), fields.end(), [&] (auto& item) { return item.id == litType.getValue(); });
                    if (found != fields.end())
                    {
                        pickedFields.push_back(*found);
                    }
                }
            };

            if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
            {
                for (auto keyType : unionType.getTypes())
                {
                    pickTypesProcessKey(keyType);
                }
            }
            else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
            {
                pickTypesProcessKey(litType);
            }
        }

        return getTupleType(pickedFields);
    }

    mlir::Type OmitTypes(mlir::Type type, mlir::Type keys)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Omit: " << type << ", keys: " << keys << "\n";);

        SmallVector<mlir_ts::FieldInfo> pickedFields;

        SmallVector<mlir_ts::FieldInfo> fields;

        std::function<boolean(mlir_ts::FieldInfo& fieldInfo, mlir::Type keys)> existKey;
        existKey = [&](mlir_ts::FieldInfo& fieldInfo, mlir::Type keys)
        {
            // get string
            if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
            {
                for (auto keyType : unionType.getTypes())
                {
                    if (existKey(fieldInfo, keyType))
                    {
                        return true;
                    }
                }
            }
            else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
            {
                return fieldInfo.id == litType.getValue();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            return false;
        };

        if (mlir::succeeded(mth.getFields(type, fields)))
        {
            for (auto& field : fields)
            {
                if (!existKey(field, keys))
                {
                    pickedFields.push_back(field);
                }
            }
        }

        return getTupleType(pickedFields);
    }        

    mlir::Type getTypeByTypeQuery(TypeQueryNode typeQueryAST, const GenContext &genContext)
    {
        auto exprName = typeQueryAST->exprName;
        if (exprName == SyntaxKind::QualifiedName)
        {
            // TODO: it seems namespace access, can u optimize it somehow?
            auto result = mlirGen(exprName.as<QualifiedName>(), genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Type();
            }

            auto val = V(result);
            return val.getType();
        }

        auto type = evaluate(exprName.as<Expression>(), genContext);
        return type;
    }

    mlir::Type getTypePredicateType(TypePredicateNode typePredicateNode, const GenContext &genContext)
    {
        auto type = getType(typePredicateNode->type, genContext);
        if (!type)
        {
            return mlir::Type();
        }

        auto namePtr = 
            typePredicateNode->parameterName == SyntaxKind::ThisType
            ? THIS_NAME
            : MLIRHelper::getName(typePredicateNode->parameterName, stringAllocator);

        // find index of parameter
        auto hasThis = false;
        auto foundParamIndex = -1;
        if (genContext.funcProto)
        {
            for (auto [index, param] : enumerate(genContext.funcProto->getParams()))
            {
                if (foundParamIndex == -1 && param->getName() == namePtr)
                {
                    foundParamIndex = index;
                }

                hasThis |= param->getName() == THIS_NAME;
            }
        }

        auto parametereNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), namePtr);
        return mlir_ts::TypePredicateType::get(parametereNameSymbol, type, !!typePredicateNode->assertsModifier, foundParamIndex - (hasThis ? 1 : 0));
    }

    mlir::Type processConditionalForType(ConditionalTypeNode conditionalTypeNode, mlir::Type checkType, mlir::Type extendsType, mlir::Type inferType, const GenContext &genContext)
    {
        auto &typeParamsWithArgs = const_cast<GenContext &>(genContext).typeParamsWithArgs;

        auto location = loc(conditionalTypeNode);

        mlir::Type resType;
        auto extendsResult = mth.extendsType(location, checkType, extendsType, typeParamsWithArgs);
        if (extendsResult == ExtendsResult::Never)
        {
            return getNeverType();
        }

        if (isTrue(extendsResult))
        {
            if (inferType)
            {
                auto namedGenType = mlir::cast<mlir_ts::NamedGenericType>(inferType);
                auto typeParam = std::make_shared<TypeParameterDOM>(namedGenType.getName().getValue().str());
                zipTypeParameterWithArgument(location, typeParamsWithArgs, typeParam, checkType, false, genContext, false);
            }

            resType = getType(conditionalTypeNode->trueType, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [TRUE] = " << resType << "\n";);

            if (extendsResult != ExtendsResult::Any)
            {
                // in case of any we need "union" of true & false
                return resType;
            }
        }

        // false case
        if (inferType)
        {
            auto namedGenType = mlir::cast<mlir_ts::NamedGenericType>(inferType);
            auto typeParam = std::make_shared<TypeParameterDOM>(namedGenType.getName().getValue().str());
            zipTypeParameterWithArgument(location, typeParamsWithArgs, typeParam, checkType, false, genContext, false);
        }

        auto falseType = getType(conditionalTypeNode->falseType, genContext);

        if (extendsResult != ExtendsResult::Any || !resType)
        {
            resType = falseType;
            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [FALSE] = " << resType << "\n";);
        }
        else
        {
            resType = getUnionType(location, resType, falseType);
            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [TRUE | FALSE] = " << resType << "\n";);
        }

        return resType;
    }

    mlir::Type getConditionalType(ConditionalTypeNode conditionalTypeNode, const GenContext &genContext)
    {
        auto checkType = getType(conditionalTypeNode->checkType, genContext);
        auto extendsType = getType(conditionalTypeNode->extendsType, genContext);
        if (!checkType || !extendsType)
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! condition type check: " << checkType << ", extends: " << extendsType << "\n";);

        if (isa<mlir_ts::NamedGenericType>(checkType) || isa<mlir_ts::NamedGenericType>(extendsType))
        {
            // we do not need to resolve it, it is generic
            auto trueType = getType(conditionalTypeNode->trueType, genContext);
            auto falseType = getType(conditionalTypeNode->falseType, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type, check: " << checkType << " extends: " << extendsType << " true: " << trueType << " false: " << falseType << " \n";);

            return getConditionalType(checkType, extendsType, trueType, falseType);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(checkType))
        {
            // we need to have original type to infer types from union
            GenContext noTypeArgsContext(genContext);
            llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsOnly;
            for (auto &pair : noTypeArgsContext.typeParamsWithArgs)
            {
                typeParamsOnly[pair.getKey()] = std::make_pair(std::get<0>(pair.getValue()), getNamedGenericType(pair.getKey()));
            }

            noTypeArgsContext.typeParamsWithArgs = typeParamsOnly;

            auto originalCheckType = getType(conditionalTypeNode->checkType, noTypeArgsContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! check type: " << checkType << " original: " << originalCheckType << " \n";);

            SmallVector<mlir::Type> results;
            for (auto subType : unionType.getTypes())
            {
                auto resSubType = processConditionalForType(conditionalTypeNode, subType, extendsType, originalCheckType, genContext);
                if (!resSubType)
                {
                    return mlir::Type();
                }

                if (resSubType != getNeverType())
                {
                    results.push_back(resSubType);
                }
            }            

            return getUnionType(results);
        }

        return processConditionalForType(conditionalTypeNode, checkType, extendsType, mlir::Type(), genContext);
    }

    mlir::Type getKeyOf(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        auto location = loc(typeOperatorNode);

        auto type = getType(typeOperatorNode->type, genContext);
        if (!type)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! can't take 'keyof'\n";);
            emitError(location, "can't take keyof");
            return mlir::Type();
        }

        return getKeyOf(location, type, genContext);
    }

    mlir::Type getKeyOf(mlir::Location location, mlir::Type type, const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! 'keyof' from: " << type << "\n";);

        if (isa<mlir_ts::AnyType>(type))
        {
            // TODO: and all methods etc
            return getUnionType(location, getStringType(), getNumberType());
        }

        if (isa<mlir_ts::UnknownType>(type))
        {
            // TODO: should be the same as Any?
            return getNeverType();
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (isa<mlir_ts::StringType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto objType = dyn_cast<mlir_ts::ObjectType>(type))
        {
            // TODO: I think this is mistake
            type = objType.getStorageType();
        }

        if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto subType : unionType.getTypes())
            {
                auto keyType = getKeyOf(location, subType, genContext);
                literalTypes.push_back(keyType);
            }

            return getUnionType(literalTypes);
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(type))
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto dictValuePair : enumType.getValues())
            {
                auto litType = mlir_ts::LiteralType::get(builder.getStringAttr(dictValuePair.getName().str()), getStringType());
                literalTypes.push_back(litType);
            }

            return getUnionType(literalTypes);
        }

        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(type))
        {
            return getKeyOfType(namedGenericType);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't take 'keyof' from: " << type << "\n";);

        emitError(location, "can't take keyof: ") << to_print(type);

        return mlir::Type();
    }

    mlir::Type getTypeOperator(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        if (typeOperatorNode->_operator == SyntaxKind::UniqueKeyword)
        {
            // TODO: finish it
            return getType(typeOperatorNode->type, genContext);
        }
        else if (typeOperatorNode->_operator == SyntaxKind::KeyOfKeyword)
        {
            return getKeyOf(typeOperatorNode, genContext);
        }
        else if (typeOperatorNode->_operator == SyntaxKind::ReadonlyKeyword)
        {
            // TODO: finish it
            return getType(typeOperatorNode->type, genContext);
        }        

        llvm_unreachable("not implemented");
    }

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::ArrayType type)
    {
        return type.getElementType();
    }

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::ConstArrayType type)
    {
        return type.getElementType();
    }

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::StringType type)
    {
        return getCharType();
    }

    template<typename T> mlir::Type getIndexedAccessTypeForArray(T type, mlir::Type indexType, const GenContext &genContext)
    {
        auto effectiveIndexType = indexType;
        if (auto litIndexType = dyn_cast<mlir_ts::LiteralType>(effectiveIndexType))
        {
            if (auto strAttr = dyn_cast<mlir::StringAttr>(litIndexType.getValue()))
            {
                if (strAttr.getValue() == LENGTH_FIELD_NAME)
                {
                    return getNumberType();
                }
            }

            effectiveIndexType = litIndexType.getElementType();
        }

        if (isa<mlir_ts::NumberType>(effectiveIndexType) || effectiveIndexType.isIntOrIndexOrFloat())
        {
            return getIndexedAccessTypeForArrayElement(type);
        }

        return mlir::Type();
    }

    // TODO: sync it with mth.getFields
    mlir::Type getIndexedAccessType(mlir::Type type, mlir::Type indexType, const GenContext &genContext)
    {
        // in case of Generic Methods but not specialized yet
        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(type))
        {
            return getIndexAccessType(type, indexType);
        }

        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(indexType))
        {
            return getIndexAccessType(type, indexType);
        }

        if (isa<mlir_ts::StringType>(indexType))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! IndexedAccessType for : " << type << " index " << indexType << " is not implemeneted, index type should not be 'string' it should be literal type \n";);
            llvm_unreachable("not implemented");
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return getIndexedAccessType(literalType.getElementType(), indexType, genContext);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> types;
            for (auto subType : unionType)
            {
                auto typeByKey = getIndexedAccessType(subType, indexType, genContext);
                if (!typeByKey)
                {
                    return mlir::Type();
                }

                types.push_back(typeByKey);
            }

            return getUnionType(types);
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(indexType))
        {
            SmallVector<mlir::Type> resolvedTypes;
            for (auto itemType : unionType.getTypes())
            {
                auto resType = getIndexedAccessType(type, itemType, genContext);
                if (!resType)
                {
                    return mlir::Type();
                }

                resolvedTypes.push_back(resType);
            }

            return getUnionType(resolvedTypes);
        }

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
        {
            // TODO: rewrite using mth.getFieldTypeByIndex(type, indexType);
            return getIndexedAccessTypeForArray(arrayType, indexType, genContext);
        }

        if (auto arrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            return getIndexedAccessTypeForArray(arrayType, indexType, genContext);
        }

        if (auto stringType = dyn_cast<mlir_ts::StringType>(type))
        {
            return getIndexedAccessTypeForArray(stringType, indexType, genContext);
        }

        if (auto objType = dyn_cast<mlir_ts::ObjectType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        // TODO: sync it with mth.getFields
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto anyType = dyn_cast<mlir_ts::AnyType>(type))
        {
            return anyType;
        }

        if (isa<mlir_ts::NeverType>(type))
        {
            return type;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! IndexedAccessType for : \n\t" << type << " \n\tindex " << indexType << " is not implemeneted \n";);

        llvm_unreachable("not implemented");
        //return mlir::Type();
    }

    mlir::Type getIndexedAccessType(IndexedAccessTypeNode indexedAccessTypeNode, const GenContext &genContext)
    {
        auto type = getType(indexedAccessTypeNode->objectType, genContext);
        if (!type)
        {
            return type;
        }

        auto indexType = getType(indexedAccessTypeNode->indexType, genContext);
        if (!indexType)
        {
            return indexType;
        }

        return getIndexedAccessType(type, indexType, genContext);
    }

    mlir::Type getTemplateLiteralType(TemplateLiteralTypeNode templateLiteralTypeNode, const GenContext &genContext)
    {
        auto location = loc(templateLiteralTypeNode);

        // first string
        auto text = convertWideToUTF8(templateLiteralTypeNode->head->rawText);

        SmallVector<mlir::Type> types;
        getTemplateLiteralSpan(types, text, templateLiteralTypeNode->templateSpans, 0, genContext);

        if (types.size() == 1)
        {
            return types.front();
        }

        return getUnionType(types);
    }

    void getTemplateLiteralSpan(SmallVector<mlir::Type> &types, const std::string &head,
                                NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex, const GenContext &genContext)
    {
        if (spanIndex >= spans.size())
        {
            auto newLiteralType = mlir_ts::LiteralType::get(builder.getStringAttr(head), getStringType());
            types.push_back(newLiteralType);
            return;
        }

        auto span = spans[spanIndex];
        auto type = getType(span->type, genContext);
        getTemplateLiteralTypeItem(types, type, head, spans, spanIndex, genContext);
    }

    void getTemplateLiteralTypeItem(SmallVector<mlir::Type> &types, mlir::Type type, const std::string &head,
                                    NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                    const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! TemplateLiteralType, processing type: " << type << ", span: " << spanIndex
                                << "\n";);

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            getTemplateLiteralUnionType(types, unionType, head, spans, spanIndex, genContext);
            return;
        }

        auto span = spans[spanIndex];

        std::stringstream ss;
        ss << head;

        auto typeText = mlir::cast<mlir::StringAttr>(mlir::cast<mlir_ts::LiteralType>(type).getValue()).getValue();
        ss << typeText.str();

        auto spanText = convertWideToUTF8(span->literal->rawText);
        ss << spanText;

        getTemplateLiteralSpan(types, ss.str(), spans, spanIndex + 1, genContext);
    }

    void getTemplateLiteralUnionType(SmallVector<mlir::Type> &types, mlir::Type unionType, const std::string &head,
                                     NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                     const GenContext &genContext)
    {
        for (auto unionTypeItem : mlir::cast<mlir_ts::UnionType>(unionType).getTypes())
        {
            getTemplateLiteralTypeItem(types, unionTypeItem, head, spans, spanIndex, genContext);
        }
    }

    mlir::Type getMappedType(MappedTypeNode mappedTypeNode, const GenContext &genContext)
    {
        // PTR(Node) /**ReadonlyToken | PlusToken | MinusToken*/ readonlyToken;
        // PTR(TypeParameterDeclaration) typeParameter;
        // PTR(TypeNode) nameType;
        // PTR(Node) /**QuestionToken | PlusToken | MinusToken*/ questionToken;
        // PTR(TypeNode) type;

        auto typeParam = processTypeParameter(mappedTypeNode->typeParameter, genContext);
        auto hasNameType = !!mappedTypeNode->nameType;

        auto constrainType = getType(typeParam->getConstraint(), genContext);
        if (!constrainType)
        {
            return mlir::Type();
        }

        if (auto keyOfType = dyn_cast<mlir_ts::KeyOfType>(constrainType))
        {
            auto type = getType(mappedTypeNode->type, genContext);
            auto nameType = getType(mappedTypeNode->nameType, genContext);
            if (!type || hasNameType && !nameType)
            {
                return mlir::Type();
            }

            return getMappedType(type, nameType, constrainType);
        }

        auto processKeyItem = [&] (mlir::SmallVector<mlir_ts::FieldInfo> &fields, mlir::Type typeParamItem) {
            const_cast<GenContext &>(genContext)
                .typeParamsWithArgs.insert({typeParam->getName(), std::make_pair(typeParam, typeParamItem)});

            auto type = getType(mappedTypeNode->type, genContext);
            if (!type)
            {
                // TODO: do we need to return error?
                // finish it
                return;
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                return; 
            }

            mlir::Type nameType = typeParamItem;
            if (hasNameType)
            {
                nameType = getType(mappedTypeNode->nameType, genContext);
            }

            // remove type param
            const_cast<GenContext &>(genContext).typeParamsWithArgs.erase(typeParam->getName());

            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... \n\t type param: [" << typeParam->getName()
                                    << " \n\t\tconstraint item: " << typeParamItem << ", \n\t\tname: " << nameType
                                    << "] \n\ttype: " << type << "\n";);

            if (mth.isNoneType(nameType) || isa<mlir_ts::NeverType>(nameType) || mth.isEmptyTuple(nameType))
            {
                // filterting out
                LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... filtered.\n";);
                return;
            }

            if (auto literalType = dyn_cast<mlir_ts::LiteralType>(nameType))
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... name: " << literalType << " type: " << type << "\n";);
                fields.push_back({literalType.getValue(), type});
            }
            else
            {
                auto nameSubType = dyn_cast<mlir_ts::UnionType>(nameType);
                auto subType = dyn_cast<mlir_ts::UnionType>(type);
                if (nameSubType && subType)
                {
                    for (auto pair : llvm::zip(nameSubType, subType))
                    {
                        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(std::get<0>(pair)))
                        {
                            auto mappedType = std::get<1>(pair);

                            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... name: " << literalType << " type: " << mappedType << "\n";);
                            fields.push_back({literalType.getValue(), mappedType});
                        }
                        else
                        {
                            llvm_unreachable("not implemented");
                        }
                    }
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
        };

        SmallVector<mlir_ts::FieldInfo> fields;
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(constrainType))
        {
            for (auto typeParamItem : unionType.getTypes())
            {
                processKeyItem(fields, typeParamItem);
            }
        }
        else if (auto litType = dyn_cast<mlir_ts::LiteralType>(constrainType))
        {
            processKeyItem(fields, litType);
        }

        if (fields.size() == 0)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type is empty for constrain: " << constrainType << ".\n";);
            emitWarning(loc(mappedTypeNode), "mapped type is empty for constrain: ")  << constrainType;
        }

        return getTupleType(fields);            
    }

    mlir_ts::VoidType getVoidType()
    {
        return mlir_ts::VoidType::get(builder.getContext());
    }

    mlir_ts::ByteType getByteType()
    {
        return mlir_ts::ByteType::get(builder.getContext());
    }

    mlir_ts::BooleanType getBooleanType()
    {
        return mlir_ts::BooleanType::get(builder.getContext());
    }

    mlir_ts::NumberType getNumberType()
    {
        return mlir_ts::NumberType::get(builder.getContext());
    }

    mlir_ts::BigIntType getBigIntType()
    {
        return mlir_ts::BigIntType::get(builder.getContext());
    }

    mlir::IndexType getIndexType()
    {
        return mlir::IndexType::get(builder.getContext());
    }

    mlir_ts::StringType getStringType()
    {
        return mlir_ts::StringType::get(builder.getContext());
    }

    mlir_ts::CharType getCharType()
    {
        return mlir_ts::CharType::get(builder.getContext());
    }

    mlir_ts::EnumType getEnumType()
    {
        return mlir_ts::EnumType::get(
            mlir::FlatSymbolRefAttr::get(builder.getContext(), StringRef{}), 
            builder.getI32Type(), 
            {});
    }

    mlir_ts::EnumType getEnumType(mlir::FlatSymbolRefAttr name, mlir::Type elementType, mlir::DictionaryAttr values)
    {
        return mlir_ts::EnumType::get(name, elementType ? elementType : builder.getI32Type(), values);
    }

    mlir_ts::ObjectStorageType getObjectStorageType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::ObjectStorageType::get(builder.getContext(), name);
    }

    mlir_ts::ClassStorageType getClassStorageType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::ClassStorageType::get(builder.getContext(), name);
    }

    mlir_ts::ClassType getClassType(mlir::FlatSymbolRefAttr name, mlir::Type storageType)
    {
        return mlir_ts::ClassType::get(name, storageType);
    }

    mlir_ts::NamespaceType getNamespaceType(mlir::StringRef name)
    {
        auto nsNameAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
        return mlir_ts::NamespaceType::get(nsNameAttr);
    }

    mlir_ts::InterfaceType getInterfaceType(StringRef fullName)
    {
        auto interfaceFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), fullName);
        return getInterfaceType(interfaceFullNameSymbol);
    }

    mlir_ts::InterfaceType getInterfaceType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::InterfaceType::get(name);
    }

    mlir::Type getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getConstArrayType(type, size);
    }

    mlir::Type getConstArrayType(mlir::Type elementType, unsigned size)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir::Type getArrayType(ArrayTypeNode arrayTypeAST, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getArrayType(type);
    }

    mlir::Type getArrayType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ArrayType::get(elementType);
    }

    mlir::Type getValueRefType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ValueRefType::get(elementType);
    }

    mlir_ts::NamedGenericType getNamedGenericType(StringRef name)
    {
        return mlir_ts::NamedGenericType::get(builder.getContext(),
                                              mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    mlir_ts::InferType getInferType(mlir::Type paramType)
    {
        assert(paramType);
        return mlir_ts::InferType::get(paramType);
    }

    mlir::Type getConditionalType(mlir::Type checkType, mlir::Type extendsType, mlir::Type trueType, mlir::Type falseType)
    {
        assert(checkType);
        assert(extendsType);
        assert(trueType);
        assert(falseType);

        if (!checkType || !extendsType || !trueType || !falseType)
        {
            return mlir::Type();
        }

        return mlir_ts::ConditionalType::get(checkType, extendsType, trueType, falseType);
    }

    mlir::Type getIndexAccessType(mlir::Type index, mlir::Type indexAccess)
    {
        assert(index);
        assert(indexAccess);

        if (!index || !indexAccess)
        {
            return mlir::Type();
        }

        return mlir_ts::IndexAccessType::get(index, indexAccess);
    }    

    mlir::Type getKeyOfType(mlir::Type type)
    {
        assert(type);

        if (!type)
        {
            return mlir::Type();
        }

        return mlir_ts::KeyOfType::get(type);
    }      

    mlir::Type getMappedType(mlir::Type elementType, mlir::Type nameType, mlir::Type constrainType)
    {
        assert(elementType);
        assert(nameType);
        assert(constrainType);

        if (!elementType || !nameType || !constrainType)
        {
            return mlir::Type();
        }

        return mlir_ts::MappedType::get(elementType, nameType, constrainType);
    }    

    mlir_ts::TypeReferenceType getTypeReferenceType(mlir::StringRef nameRef, mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::TypeReferenceType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), nameRef), types);
    }    

    mlir::Value getUndefined(mlir::Location location)
    {
        return builder.create<mlir_ts::UndefOp>(location, getUndefinedType());
    }

    mlir::Value getInfinity(mlir::Location location)
    {
#ifdef NUMBER_F64
        union { double dbl; int64_t int64; } val{};
        val.int64 = 0x7FF0000000000000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(val.dbl));
#else
        union { float flt; int32_t int32; } val;
        val.int32 = 0x7FF00000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(val.int32));
#endif
    }

    mlir::Value getNaN(mlir::Location location)
    {
#ifdef NUMBER_F64
        union { double dbl; int64_t int64; } val{};
        val.int64 = 0x7FF0000000000001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(val.dbl));
#else
        union { float flt; int32_t int32; } val;
        val.int32 = 0x7FF00001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(val.int32));
#endif
    }

    std::pair<mlir::Attribute, mlir::LogicalResult> getNameFromComputedPropertyName(Node name, const GenContext &genContext)
    {
        if (name == SyntaxKind::ComputedPropertyName)
        {
            MLIRCodeLogic mcl(builder);
            auto result = mlirGen(name.as<ComputedPropertyName>(), genContext);
            auto value = V(result);
            LLVM_DEBUG(llvm::dbgs() << "!! ComputedPropertyName: " << value << "\n";);
            auto attr = mcl.ExtractAttr(value);
            if (!attr)
            {
                emitError(loc(name), "not supported 'Computed Property Name' expression");
            }

            return {attr, attr ? mlir::success() : mlir::failure()};
        }

        return {mlir::Attribute(), mlir::success()};
    }

    mlir::Attribute TupleFieldName(Node name, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(name, stringAllocator);
        if (namePtr.empty())
        {
            auto [attrComputed, attrResult] = getNameFromComputedPropertyName(name, genContext);
            if (attrComputed || mlir::failed(attrResult))
            {
                return attrComputed;
            }
                        
            MLIRCodeLogic mcl(builder);
            auto result = mlirGen(name.as<Expression>(), genContext);
            auto value = V(result);
            auto attr = mcl.ExtractAttr(value);
            if (!attr)
            {
                emitError(loc(name), "not supported name");
            }

            return attr;
        }

        return MLIRHelper::TupleFieldName(namePtr, builder.getContext());
    }

    std::pair<bool, mlir::LogicalResult> getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        mlir::Attribute attrVal;
        auto arrayMode = true;
        auto index = 0;
        for (auto typeItem : tupleType->elements)
        {
            if (typeItem == SyntaxKind::NamedTupleMember)
            {
                auto namedTupleMember = typeItem.as<NamedTupleMember>();

                auto type = getType(namedTupleMember->type, genContext);
                if (!type)
                {
                    return {arrayMode, mlir::failure()};
                }

                types.push_back({TupleFieldName(namedTupleMember->name, genContext), type});
                arrayMode = false;
            }
            else if (typeItem == SyntaxKind::LiteralType)
            {
                auto literalTypeNode = typeItem.as<LiteralTypeNode>();
                auto result = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
                if (result.failed_or_no_value())
                {
                    return {arrayMode, mlir::failure()};
                }

                auto literalValue = V(result);
                auto constantOp = literalValue.getDefiningOp<mlir_ts::ConstantOp>();

                assert(constantOp);
                attrVal = constantOp.getValueAttr();

                if (arrayMode)
                {
                    types.push_back({builder.getIntegerAttr(builder.getI32Type(), index), constantOp.getType()});
                }

                index++;
                continue;
            }
            else
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return {arrayMode, mlir::failure()};
                }

                types.push_back({attrVal, type});
            }

            attrVal = mlir::Attribute();
        }

        return {arrayMode, mlir::success()};
    }

    mlir::LogicalResult getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        for (auto typeItem : typeLiteral->members)
        {
            SyntaxKind kind = typeItem;
            if (kind == SyntaxKind::PropertySignature)
            {
                auto propertySignature = typeItem.as<PropertySignature>();

                auto originalType = getType(propertySignature->type, genContext);
                if (!originalType)
                {
                    return mlir::failure();
                }

                auto type = mcl.getEffectiveFunctionTypeForTupleField(originalType);

                assert(type);
                types.push_back({TupleFieldName(propertySignature->name, genContext), type});
            }
            else if (kind == SyntaxKind::MethodSignature)
            {
                auto methodSignature = typeItem.as<MethodSignature>();

                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({TupleFieldName(methodSignature->name, genContext), type});
            }
            else if (kind == SyntaxKind::ConstructSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(NEW_CTOR_METHOD_NAME, builder.getContext()), type});
            }            
            else if (kind == SyntaxKind::IndexSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(INDEX_ACCESS_GET_FIELD_NAME, builder.getContext()), mth.getIndexGetFunctionType(type)});
                types.push_back({MLIRHelper::TupleFieldName(INDEX_ACCESS_SET_FIELD_NAME, builder.getContext()), mth.getIndexSetFunctionType(type)});
            }
            else if (kind == SyntaxKind::CallSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(CALL_FIELD_NAME, builder.getContext()), type});
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return mlir::success();
    }

    mlir::Type getConstTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto [arrayMode, result] = getTupleFieldInfo(tupleType, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        return getConstTupleType(types);
    }

    mlir_ts::ConstTupleType getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ConstTupleType::get(builder.getContext(), fieldInfos);
    }

    mlir::Type getTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto [arrayMode, result] = getTupleFieldInfo(tupleType, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        if (arrayMode && types.size() == 1)
        {
            return getArrayType(types.front().type);
        }

        return getTupleType(types);
    }

    mlir::Type getTupleType(TypeLiteralNode typeLiteral, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto result = getTupleFieldInfo(typeLiteral, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        // TODO: remove the following hack
        // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
        if (types.size() == 1)
        {
            auto indexAccessName = MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, builder.getContext());
            if (types.front().id == indexAccessName)
            {
                auto [arg, res] = mth.getIndexSignatureArgumentAndResultTypes(types.front().type);
                if (auto elementTypeOfIndexSignature = arg)
                {
                    auto arrayType = getArrayType(elementTypeOfIndexSignature);
                    LLVM_DEBUG(llvm::dbgs() << "\n!! this is array type: " << arrayType << "\n";);
                    return arrayType;
                }
            }
        }

        // == TODO: remove the following hack
        // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
        if (types.size() == 2)
        {
            mlir::Type indexSignatureType;
            auto lengthName = MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, builder.getContext());
            auto indexAccessName = MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, builder.getContext());
            if (types.front().id == lengthName && types.back().id == indexAccessName)
            {
                indexSignatureType = types.back().type;
            }
            
            if (types.back().id == lengthName && types.front().id == indexAccessName)
            {
                indexSignatureType = types.front().type;
            }

            if (indexSignatureType)
            {
                // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
                auto [arg, res] = mth.getIndexSignatureArgumentAndResultTypes(indexSignatureType);
                if (auto elementTypeOfIndexSignature = arg)
                {
                    auto arrayType = getArrayType(elementTypeOfIndexSignature);
                    LLVM_DEBUG(llvm::dbgs() << "\n!! this is array type: " << arrayType << "\n";);
                    return arrayType;
                }
            }
        }        

        return getTupleType(types);
    }

    mlir::Type getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::ObjectType getObjectType(mlir::Type type)
    {
        return mlir_ts::ObjectType::get(type);
    }

    mlir_ts::OpaqueType getOpaqueType()
    {
        return mlir_ts::OpaqueType::get(builder.getContext());
    }    

    mlir_ts::BoundFunctionType getBoundFunctionType(mlir_ts::FunctionType funcType)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), funcType);
    }

    mlir_ts::BoundFunctionType getBoundFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                                    bool isVarArg)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::FunctionType getFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                          bool isVarArg)
    {
        return mlir_ts::FunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::ExtensionFunctionType getExtensionFunctionType(mlir_ts::FunctionType funcType)
    {
        return mlir_ts::ExtensionFunctionType::get(builder.getContext(), funcType);
    }

    mlir::Type getSignature(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        GenContext genericTypeGenContext(genContext);

        // preparing generic context to resolve types
        if (signature->typeParameters.size())
        {
            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(
                    processTypeParameters(signature->typeParameters, typeParameters, genericTypeGenContext)))
            {
                return mlir::Type();
            }

            auto [result, hasAnyNamedGenericType] =
                zipTypeParametersWithArguments(loc(signature), typeParameters, signature->typeArguments,
                                               genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

            if (mlir::failed(result))
            {
                return mlir::Type();
            }
        }

        auto resultType = getType(signature->type, genericTypeGenContext);
        if (!resultType && !genContext.allowPartialResolve)
        {
            return mlir::Type();
        }

        SmallVector<mlir::Type> argTypes;
        auto isVarArg = false;
        for (auto paramItem : signature->parameters)
        {
            auto type = getType(paramItem->type, genericTypeGenContext);
            if (!type)
            {
                return mlir::Type();
            }

            if (paramItem->questionToken)
            {
                type = getOptionalType(type);
            }

            argTypes.push_back(type);

            isVarArg |= !!paramItem->dotDotDotToken;
        }

        auto funcType = mlir_ts::FunctionType::get(builder.getContext(), argTypes, resultType, isVarArg);
        return funcType;
    }

    mlir::Type getFunctionType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), mlir::cast<mlir_ts::FunctionType>(signatureType));
        return funcType;
    }

    mlir::Type getConstructorType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::ConstructFunctionType::get(
            builder.getContext(), 
            mlir::cast<mlir_ts::FunctionType>(signatureType), 
            hasModifier(signature, SyntaxKind::AbstractKeyword));
        return funcType;
    }

    mlir::Type getCallSignature(CallSignatureDeclaration signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), mlir::cast<mlir_ts::FunctionType>(signatureType));
        return funcType;
    }

    mlir::Type getConstructSignature(ConstructSignatureDeclaration constructSignature,
                                                const GenContext &genContext)
    {
        return getSignature(constructSignature, genContext);
    }

    mlir::Type getMethodSignature(MethodSignature methodSignature, const GenContext &genContext)
    {
        return getSignature(methodSignature, genContext);
    }

    mlir::Type getIndexSignature(IndexSignatureDeclaration indexSignature, const GenContext &genContext)
    {
        return getSignature(indexSignature, genContext);
    }

    mlir::Type getUnionType(UnionTypeNode unionTypeNode, const GenContext &genContext)
    {
        MLIRTypeHelper::UnionTypeProcessContext unionContext = {};
        for (auto typeItem : unionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! wrong type: " << loc(typeItem) << "\n";);

                //llvm_unreachable("wrong type");
                return mlir::Type();
            }

            mth.processUnionTypeItem(type, unionContext);
        }

        // default wide types
        if (unionContext.isAny)
        {
            return getAnyType();
        }

        return mth.getUnionTypeMergeTypes(loc(unionTypeNode), unionContext, false, false);
    }

    mlir::Type getUnionType(mlir::Location location, mlir::Type type1, mlir::Type type2)
    {
        if (mth.isNoneType(type1) || mth.isNoneType(type2))
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! join: " << type1 << " | " << type2;);

        auto resType = mth.getUnionType(location, type1, type2, false);

        LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

        return resType;
    }

    mlir::Type getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mth.getUnionType(types);
    }

    mlir::LogicalResult processIntersectionType(InterfaceInfo::TypePtr newInterfaceInfo, mlir::Type type, bool conditional = false)
    {
        if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            auto srcInterfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
            assert(srcInterfaceInfo);
            newInterfaceInfo->extends.push_back({-1, srcInterfaceInfo});
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            mergeInterfaces(newInterfaceInfo, tupleType, conditional);
        }
        else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            mergeInterfaces(newInterfaceInfo, mlir::cast<mlir_ts::TupleType>(mth.removeConstType(constTupleType)), conditional);
        }              
        else if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto type : unionType.getTypes())
            {
                if (mlir::failed(processIntersectionType(newInterfaceInfo, type, true)))
                {
                    return mlir::failure();
                }
            }            
        }              
        else
        {
            return mlir::failure();
        }      

        return mlir::success();
    }

    mlir::Type getIntersectionType(IntersectionTypeNode intersectionTypeNode, const GenContext &genContext)
    {
        mlir_ts::InterfaceType baseInterfaceType;
        mlir_ts::TupleType baseTupleType;
        mlir::SmallVector<mlir::Type> types;
        mlir::SmallVector<mlir::Type> typesForUnion;
        auto allTupleTypesConst = true;
        auto unionTypes = false;
        for (auto typeItem : intersectionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                return mlir::Type();
            }

            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                allTupleTypesConst = false;
                if (!baseTupleType)
                {
                    baseTupleType = tupleType;
                }
            }

            if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                if (!baseTupleType)
                {
                    baseTupleType = mlir_ts::TupleType::get(builder.getContext(), constTupleType.getFields());
                }
            }

            if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(type))
            {
                if (!baseInterfaceType)
                {
                    baseInterfaceType = ifaceType;
                }
            }

            types.push_back(type);
        }

        if (types.size() == 0)
        {
            // this is never type
            return getNeverType();
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        // find base type
        if (baseInterfaceType)
        {
            auto declareInterface = false;
            auto newInterfaceInfo = newInterfaceType(intersectionTypeNode, declareInterface, genContext);
            if (declareInterface)
            {
                // merge all interfaces;
                for (auto type : types)
                {
                    if (mlir::failed(processIntersectionType(newInterfaceInfo, type)))
                    {
                        emitWarning(loc(intersectionTypeNode), "Intersection can't be resolved.");
                        return getIntersectionType(types);
                    }
                }
            }

            newInterfaceInfo->recalcOffsets();

            return newInterfaceInfo->interfaceType;
        }

        if (baseTupleType)
        {
            auto anyTypesInBaseTupleType = baseTupleType.getFields().size() > 0;

            SmallVector<::mlir::typescript::FieldInfo> typesForNewTuple;
            for (auto type : types)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! processing ... & {...} :" << type << "\n";);

                // umwrap optional
                if (!anyTypesInBaseTupleType)
                {
                    type = mth.stripOptionalType(type);
                }

                if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
                {
                    allTupleTypesConst = false;
                    for (auto field : tupleType.getFields())
                    {
                        typesForNewTuple.push_back(field);
                    }
                }
                else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
                {
                    for (auto field : constTupleType.getFields())
                    {
                        typesForNewTuple.push_back(field);
                    }
                }
                else if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
                {
                    if (!anyTypesInBaseTupleType)
                    {
                        unionTypes = true;
                        for (auto subType : unionType.getTypes())
                        {
                            if (subType == getNullType() || subType == getUndefinedType())
                            {
                                continue;
                            }

                            typesForUnion.push_back(subType);
                        }
                    }                    
                }
                else
                {
                    if (!anyTypesInBaseTupleType)
                    {
                        unionTypes = true; 
                        typesForUnion.push_back(type);
                    }
                    else
                    {
                        // no intersection
                        return getNeverType();
                    }
                }
            }

            if (unionTypes)
            {
                auto resUnion = getUnionType(typesForUnion);
                LLVM_DEBUG(llvm::dbgs() << "\n!! &=: " << resUnion << "\n";);
                return resUnion;                
            }

            auto resultType = allTupleTypesConst 
                ? (mlir::Type)getConstTupleType(typesForNewTuple)
                : (mlir::Type)getTupleType(typesForNewTuple);

            LLVM_DEBUG(llvm::dbgs() << "\n!! &=: " << resultType << "\n";);

            return resultType;
        }

        // calculate of intersection between types and literal types
        mlir::Type resType;
        for (auto typeItem : types)
        {
            if (!resType)
            {
                resType = typeItem;
                continue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! &: " << resType << " & " << typeItem;);

            resType = AndType(resType, typeItem);

            LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

            if (isa<mlir_ts::NeverType>(resType))
            {
                return getNeverType();
            }
        }

        if (resType)
        {
            return resType;
        }

        return getNeverType();
    }

    mlir::Type getIntersectionType(mlir::Type type1, mlir::Type type2)
    {
        if (!type1 || !type2)
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! intersection: " << type1 << " & " << type2;);

        auto resType = mth.getIntersectionType(type1, type2);

        LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

        return resType;
    }

    mlir::Type getIntersectionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mth.getIntersectionType(types);
    }

    mlir::Type AndType(mlir::Type left, mlir::Type right)
    {
        // TODO: 00types_unknown1.ts contains examples of results with & | for types,  T & {} == T & {}, T | {} == T |
        // {}, (they do not change)
        if (left == right)
        {
            return left;
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(right))
        {
            if (literalType.getElementType() == left)
            {
                if (isa<mlir_ts::LiteralType>(left))
                {
                    return getNeverType();
                }

                return literalType;
            }
        }

        if (auto leftUnionType = dyn_cast<mlir_ts::UnionType>(left))
        {
            return AndUnionType(leftUnionType, right);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(right))
        {
            mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
            for (auto unionTypeItem : unionType.getTypes())
            {
                auto resType = AndType(left, unionTypeItem);
                newUniqueTypes.insert(resType);
            }

            SmallVector<mlir::Type> newTypes;
            for (auto uniqType : newUniqueTypes)
            {
                newTypes.push_back(uniqType);
            }

            return getUnionType(newTypes);
        }

        if (isa<mlir_ts::NullType>(left))
        {

            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (isa<mlir_ts::NullType>(right))
        {

            if (mth.isValueType(left))
            {
                return getNeverType();
            }

            return right;
        }

        if (isa<mlir_ts::NullType>(left))
        {

            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (isa<mlir_ts::AnyType>(left) || isa<mlir_ts::UnknownType>(left))
        {
            return right;
        }

        if (isa<mlir_ts::AnyType>(right) || isa<mlir_ts::UnknownType>(right))
        {
            return left;
        }

        // TODO: should I add, interface, tuple types here?
        // PS: string & { __b: number } creating type "string & { __b: number }".

        return getIntersectionType(left, right);
    }

    mlir::Type AndUnionType(mlir_ts::UnionType leftUnion, mlir::Type right)
    {
        mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
        for (auto unionTypeItem : leftUnion.getTypes())
        {
            auto resType = AndType(unionTypeItem, right);
            newUniqueTypes.insert(resType);
        }

        SmallVector<mlir::Type> newTypes;
        for (auto uniqType : newUniqueTypes)
        {
            newTypes.push_back(uniqType);
        }

        return getUnionType(newTypes);
    }

    InterfaceInfo::TypePtr newInterfaceType(IntersectionTypeNode intersectionTypeNode, bool &declareInterface,
                                            const GenContext &genContext)
    {
        auto newName = MLIRHelper::getAnonymousName(loc_check(intersectionTypeNode), "ifce", "");

        // clone into new interface
        auto interfaceInfo = mlirGenInterfaceInfo(newName, declareInterface, genContext);

        return interfaceInfo;
    }

    mlir::LogicalResult mergeInterfaces(InterfaceInfo::TypePtr dest, mlir_ts::TupleType src, bool conditional = false)
    {
        // TODO: use it to merge with TupleType
        for (auto &item : src.getFields())
        {
            dest->fields.push_back({item.id, item.type, item.isConditional || conditional, dest->getNextVTableMemberIndex()});
        }

        return mlir::success();
    }

    mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode, const GenContext &genContext)
    {
        return getType(parenthesizedTypeNode->type, genContext);
    }

    mlir::Type getLiteralType(LiteralTypeNode literalTypeNode)
    {
        GenContext genContext{};
        genContext.dummyRun = true;
        genContext.allowPartialResolve = true;
        auto result = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
        auto value = V(result);
        auto type = value.getType();

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return literalType;
        }

        auto constantOp = value.getDefiningOp<mlir_ts::ConstantOp>();
        if (constantOp)
        {
            auto valueAttr = value.getDefiningOp<mlir_ts::ConstantOp>().getValueAttr();
            auto literalType = mlir_ts::LiteralType::get(valueAttr, type);
            return literalType;
        }

        auto nullOp = value.getDefiningOp<mlir_ts::NullOp>();
        if (nullOp)
        {
            return getNullType();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! value of literal: " << value << "\n";);

        llvm_unreachable("not implemented");
    }

    mlir::Type getOptionalType(OptionalTypeNode optionalTypeNode, const GenContext &genContext)
    {
        return getOptionalType(getType(optionalTypeNode->type, genContext));
    }

    mlir::Type getOptionalType(mlir::Type type)
    {
        if (!type)
        {
            return mlir::Type();
        }

        if (isa<mlir_ts::OptionalType>(type))
        {
            return type;
        }        

        return mlir_ts::OptionalType::get(type);
    }

    mlir::Type getRestType(RestTypeNode restTypeNode, const GenContext &genContext)
    {
        auto arrayType = getType(restTypeNode->type, genContext);
        if (!arrayType)
        {
            return mlir::Type();
        }

        return getConstArrayType(mlir::cast<mlir_ts::ArrayType>(arrayType).getElementType(), 0);
    }

    mlir_ts::AnyType getAnyType()
    {
        return mlir_ts::AnyType::get(builder.getContext());
    }

    mlir_ts::UnknownType getUnknownType()
    {
        return mlir_ts::UnknownType::get(builder.getContext());
    }

    mlir_ts::NeverType getNeverType()
    {
        return mlir_ts::NeverType::get(builder.getContext());
    }

    mlir_ts::ConstType getConstType()
    {
        return mlir_ts::ConstType::get(builder.getContext());
    }    

    mlir_ts::SymbolType getSymbolType()
    {
        return mlir_ts::SymbolType::get(builder.getContext());
    }

    mlir_ts::UndefinedType getUndefinedType()
    {
        return mlir_ts::UndefinedType::get(builder.getContext());
    }

    mlir_ts::NullType getNullType()
    {
        return mlir_ts::NullType::get(builder.getContext());
    }

    mlir::LogicalResult declare(mlir::Location location, VariableDeclarationDOM::TypePtr var, mlir::Value value, const GenContext &genContext, bool showWarnings = false)
    {
        if (!value)
        {
            return mlir::failure();
        }

        const auto &name = var->getName();

        //LLVM_DEBUG(llvm::dbgs() << "\n!! declare variable: " << name << " = [" << value << "]\n";);

        if (showWarnings && symbolTable.count(name))
        {
            auto previousVariable = symbolTable.lookup(name).first;
            if (previousVariable.getParentBlock() == value.getParentBlock())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! WARNING redeclaration: " << name << " = [" << value << "]\n";);
                // TODO: find out why you have redeclared vars

                std::string loc;
                llvm::raw_string_ostream sloc(loc);
                printLocation(sloc, previousVariable.getLoc(), path, true);
                sloc.flush();
                emitWarning(location, "") << "variable "<< name << " redeclared. Previous declaration: " << sloc.str();                
            }
        }

        if (compileOptions.generateDebugInfo)
        {
            if (auto defOp = value.getDefiningOp())
            {
                MLIRDebugInfoHelper mti(builder, debugScope);
                defOp->setLoc(mti.combineWithCurrentScopeAndName(defOp->getLoc(), var->getName()));
            }
        }

        if (!genContext.insertIntoParentScope)
        {
            symbolTable.insert(name, {value, var});
        }
        else
        {
            symbolTable.insertIntoScope(symbolTable.getCurScope()->getParentScope(), name, {value, var});
        }

        return mlir::success();
    }

    bool isAddedToExport(mlir::Type type)
    {
        if (stage != Stages::SourceGeneration)
        {
            return true;
        }

        return exportedTypes.contains(type);
    }

    bool isExportDependencyChecked(mlir::Type type)
    {
        if (stage != Stages::SourceGeneration)
        {
            return true;
        }

        return exportCheckedDependenciesTypes.contains(type);
    }

    bool addDependancyTypesToExport(mlir::Type type)
    {
        if (isExportDependencyChecked(type))
        {
            // already added
            return true;
        }

        exportCheckedDependenciesTypes.insert(type);

        // iterate all types
        mth.forEachTypes(type, [&] (mlir::Type subType) {
            return addDependancyTypesToExport(subType);
        });

        addTypeDeclarationToExport(type);

        return false;
    }

    // base method
    bool addDependancyTypesToExportNoCheck(mlir::Type type)
    {
        auto cont = mlir::TypeSwitch<mlir::Type, bool>(type)
            .Case<mlir_ts::InterfaceType>([&](auto ifaceType) {
                auto interfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
                assert(interfaceInfo);

                for (auto& method : interfaceInfo->methods)
                {
                    addDependancyTypesToExport(method.funcType);
                }

                for (auto& field : interfaceInfo->fields)
                {
                    addDependancyTypesToExport(field.type);
                }

                return true;
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);

                for (auto& method : classInfo->methods)
                {
                    addDependancyTypesToExport(method.funcType);
                }

                for (auto& accessor : classInfo->accessors)
                {
                    if (accessor.get) addDependancyTypesToExport(accessor.get.getFunctionType());
                    if (accessor.set) addDependancyTypesToExport(accessor.set.getFunctionType());
                }                

                return true;
            })
            .Case<mlir_ts::EnumType>([&](auto enumType) {
                // no dependancies here
                return true;
            })
            .Default([&](auto type) {
                return true;
            });

        return cont;
    }

    bool addTypeDeclarationToExport(mlir::Type type)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! adding type declaration to export: \n" << type << "\n";);

        if (isAddedToExport(type))
        {
            // already added
            LLVM_DEBUG(llvm::dbgs() << "\n!! ALREADY ADDED to export: \n" << type << "\n";);
            return true;
        }        

        return addTypeDeclarationToExportNoCheck(type);
    }

    bool addTypeDeclarationToExportNoCheck(mlir::Type type)
    {
        auto cont = mlir::TypeSwitch<mlir::Type, bool>(type)
            .Case<mlir_ts::InterfaceType>([&](auto ifaceType) {
                auto interfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
                assert(interfaceInfo);
                addInterfaceDeclarationToExport(interfaceInfo);
                return true;
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);
                addClassDeclarationToExport(classInfo);
                return true;
            })
            .Case<mlir_ts::EnumType>([&](auto enumType) {
                auto enumInfo = getEnumInfoByFullName(enumType.getName().getValue());
                assert(enumInfo);
                assert(enumInfo->enumType == enumType);

                addEnumDeclarationToExport(enumInfo->name, enumInfo->elementNamespace, enumType);
                return true;
            })
            .Default([&](auto type) {
                return true;
            });

        return cont;
    }    

    void addTypeDeclarationToExport(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir::Type type)    
    {
        // TODO: add distinct declaration

        // we need to add it anyway as it is type declaration
        addDependancyTypesToExport(type);

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.printTypeDeclaration(name, elementNamespace, type);

        declExports << ss.str().str();
    }

    void addInterfaceDeclarationToExport(InterfaceInfo::TypePtr interfaceInfo)
    {
        if (isAddedToExport(interfaceInfo->interfaceType))
        {
            // already added
            return;
        }

        exportedTypes.insert(interfaceInfo->interfaceType);

        addDependancyTypesToExport(interfaceInfo->interfaceType);

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.print(interfaceInfo);

        declExports << ss.str().str();
    }

    void addEnumDeclarationToExport(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir_ts::EnumType enumType)
    {
        if (isAddedToExport(enumType))
        {
            // already added
            return;
        }        

        exportedTypes.insert(enumType);

        //addDependancyTypesToExport(enumType);

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.printEnum(name, elementNamespace, enumType.getValues());

        declExports << ss.str().str();        
    }

    void addVariableDeclarationToExport(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir::Type type, bool isConst)
    {               
        // TODO: add distinct declaration

        // we need to add it anyway as it is varaible declaration
        addDependancyTypesToExport(type);

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.printVariableDeclaration(name, elementNamespace, type, isConst);

        declExports << ss.str().str();
    }

    void addFunctionDeclarationToExport(FunctionPrototypeDOM::TypePtr funcProto, NamespaceInfo::TypePtr elementNamespace)
    {
        // TODO: add distinct declaration

        // we need to add it anyway as it is function declaration
        addDependancyTypesToExport(funcProto->getFuncType());

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.print(funcProto->getNameWithoutNamespace(), elementNamespace, funcProto->getFuncType());

        declExports << ss.str().str();
    }

    void addClassDeclarationToExport(ClassInfo::TypePtr newClassPtr)
    {
        if (isAddedToExport(newClassPtr->classType))
        {
            // already added
            return;
        }    

        exportedTypes.insert(newClassPtr->classType);

        addDependancyTypesToExport(newClassPtr->classType);

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);        
        MLIRDeclarationPrinter dp(ss);
        dp.print(newClassPtr);

        declExports << ss.str().str();
    }

    auto getNamespaceName() -> StringRef
    {
        return currentNamespace->name;
    }

    auto getFullNamespaceName() -> StringRef
    {
        return currentNamespace->fullName;
    }

    auto getFullNamespaceName(StringRef name) -> StringRef
    {
        if (currentNamespace->fullName.empty())
        {
            return StringRef(name).copy(stringAllocator);
        }

        std::string res;
        res += currentNamespace->fullName;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto getGlobalsFullNamespaceName(StringRef name) -> StringRef
    {
        auto globalsFullNamespaceName = getGlobalsNamespaceFullName();

        if (globalsFullNamespaceName.empty())
        {
            return StringRef(name).copy(stringAllocator);
        }

        std::string res;
        res += globalsFullNamespaceName;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;            
    }

    auto concat(StringRef fullNamespace, StringRef name) -> StringRef
    {
        std::string res;
        res += fullNamespace;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto concat(StringRef fullNamespace, StringRef className, StringRef name) -> StringRef
    {
        std::string res;
        res += fullNamespace;
        res += ".";
        res += className;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto concat(StringRef fullNamespace, StringRef className, StringRef name, int index) -> StringRef
    {
        std::string res;
        res += fullNamespace;
        res += ".";
        res += className;
        res += ".";
        res += name;
        res += "#";
        res += std::to_string(index);

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }    

    template <typename T> bool is_default(T &t)
    {
        return !static_cast<bool>(t);
    }

#define lookupLogic(S)                                                                                                 \
    MLIRNamespaceGuard ng(currentNamespace);                                                                           \
    decltype(currentNamespace->S.lookup(name)) res;                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        res = currentNamespace->S.lookup(name);                                                                        \
        if (!is_default(res) || !currentNamespace->isFunctionNamespace)                                                \
        {                                                                                                              \
            break;                                                                                                     \
        }                                                                                                              \
                                                                                                                       \
        currentNamespace = currentNamespace->parentNamespace;                                                          \
    } while (true);                                                                                                    \
                                                                                                                       \
    return res;

#define existLogic(S)                                                                                                  \
    MLIRNamespaceGuard ng(currentNamespace);                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto res = currentNamespace->S.count(name);                                                                    \
        if (res > 0)                                                                                                   \
        {                                                                                                              \
            return true;                                                                                               \
        }                                                                                                              \
                                                                                                                       \
        if (!currentNamespace->isFunctionNamespace)                                                                    \
        {                                                                                                              \
            return false;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        currentNamespace = currentNamespace->parentNamespace;                                                          \
    } while (true);                                                                                                    \
                                                                                                                       \
    return false;

#define removeLogic(S)                                                                                                 \
    MLIRNamespaceGuard ng(currentNamespace);                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto res = currentNamespace->S.count(name);                                                                    \
        if (res > 0)                                                                                                   \
        {                                                                                                              \
            currentNamespace->S.erase(name.str());                                                                    \
            return true;                                                                                               \
        }                                                                                                              \
                                                                                                                       \
        if (!currentNamespace->isFunctionNamespace)                                                                    \
        {                                                                                                              \
            return false;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        currentNamespace = currentNamespace->parentNamespace;                                                          \
    } while (true);                                                                                                    \
                                                                                                                       \
    return false;    

    auto getNamespaceByFullName(StringRef fullName) -> NamespaceInfo::TypePtr
    {
        return fullNamespacesMap.lookup(fullName);
    }

    auto getNamespaceMap() -> llvm::StringMap<NamespaceInfo::TypePtr> &
    {
        return currentNamespace->namespacesMap;
    }

    auto getFunctionTypeMap() -> llvm::StringMap<mlir_ts::FunctionType> &
    {
        return currentNamespace->functionTypeMap;
    }

    auto lookupFunctionTypeMap(StringRef name) -> mlir_ts::FunctionType
    {
        lookupLogic(functionTypeMap);
    }

    auto getFunctionMap() -> llvm::StringMap<mlir_ts::FuncOp> &
    {
        return currentNamespace->functionMap;
    }

    auto lookupFunctionMap(StringRef name) -> mlir_ts::FuncOp
    {
        lookupLogic(functionMap);
    }

    // TODO: all lookup/count should be replaced by GenericFunctionMapLookup
    auto getGenericFunctionMap() -> llvm::StringMap<GenericFunctionInfo::TypePtr> &
    {
        return currentNamespace->genericFunctionMap;
    }

    auto lookupGenericFunctionMap(StringRef name) -> GenericFunctionInfo::TypePtr
    {
        lookupLogic(genericFunctionMap);
    }

    auto existGenericFunctionMap(StringRef name) -> bool
    {
        existLogic(genericFunctionMap);
    }

    auto removeGenericFunctionMap(StringRef name) -> bool
    {
        removeLogic(genericFunctionMap);
    }    

    auto getGlobalsNamespaceFullName() -> llvm::StringRef
    {
        if (!currentNamespace->isFunctionNamespace)
        {
            return currentNamespace->fullName;
        }

        auto curr = currentNamespace;
        while (curr->isFunctionNamespace)
        {
            curr = curr->parentNamespace;
        }

        return curr->fullName;
    }    

    auto getGlobalsMap() -> llvm::StringMap<VariableDeclarationDOM::TypePtr> &
    {
        if (!currentNamespace->isFunctionNamespace)
        {
            return currentNamespace->globalsMap;
        }

        auto curr = currentNamespace;
        while (curr->isFunctionNamespace)
        {
            curr = curr->parentNamespace;
        }

        return curr->globalsMap;
    }

    auto getCaptureVarsMap() -> llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> &
    {
        return currentNamespace->captureVarsMap;
    }

    auto getLocalVarsInThisContextMap() -> llvm::StringMap<llvm::SmallVector<mlir::typescript::FieldInfo>> &
    {
        return currentNamespace->localVarsInThisContextMap;
    }

    template <typename T> bool is_default(llvm::SmallVector<T> &t)
    {
        return t.size() == 0;
    }

    auto lookupLocalVarsInThisContextMap(StringRef name) -> llvm::SmallVector<mlir::typescript::FieldInfo>
    {
        lookupLogic(localVarsInThisContextMap);
    }

    auto existLocalVarsInThisContextMap(StringRef name) -> bool
    {
        existLogic(localVarsInThisContextMap);
    }

    auto getClassesMap() -> llvm::StringMap<ClassInfo::TypePtr> &
    {
        return currentNamespace->classesMap;
    }

    auto getGenericClassesMap() -> llvm::StringMap<GenericClassInfo::TypePtr> &
    {
        return currentNamespace->genericClassesMap;
    }

    auto lookupGenericClassesMap(StringRef name) -> GenericClassInfo::TypePtr
    {
        lookupLogic(genericClassesMap);
    }

    auto getInterfacesMap() -> llvm::StringMap<InterfaceInfo::TypePtr> &
    {
        return currentNamespace->interfacesMap;
    }

    auto getGenericInterfacesMap() -> llvm::StringMap<GenericInterfaceInfo::TypePtr> &
    {
        return currentNamespace->genericInterfacesMap;
    }

    auto lookupGenericInterfacesMap(StringRef name) -> GenericInterfaceInfo::TypePtr
    {
        lookupLogic(genericInterfacesMap);
    }

    auto getEnumsMap() -> llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> &
    {
        return currentNamespace->enumsMap;
    }

    auto getTypeAliasMap() -> llvm::StringMap<mlir::Type> &
    {
        return currentNamespace->typeAliasMap;
    }

    auto getGenericTypeAliasMap()
        -> llvm::StringMap<std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode>> &
    {
        return currentNamespace->genericTypeAliasMap;
    }

    bool is_default(std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode> &t)
    {
        return std::get<0>(t).size() == 0;
    }

    auto lookupGenericTypeAliasMap(StringRef name) -> std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode>
    {
        lookupLogic(genericTypeAliasMap);
    }

    auto getImportEqualsMap() -> llvm::StringMap<mlir::StringRef> &
    {
        return currentNamespace->importEqualsMap;
    }

    auto getGenericFunctionInfoByFullName(StringRef fullName) -> GenericFunctionInfo::TypePtr
    {
        return fullNameGenericFunctionsMap.lookup(fullName);
    }

    auto getEnumInfoByFullName(StringRef fullName) -> EnumInfo::TypePtr
    {
        return fullNameEnumsMap.lookup(fullName);
    }

    auto getClassInfoByFullName(StringRef fullName) -> ClassInfo::TypePtr
    {
        return fullNameClassesMap.lookup(fullName);
    }

    auto getGenericClassInfoByFullName(StringRef fullName) -> GenericClassInfo::TypePtr
    {
        return fullNameGenericClassesMap.lookup(fullName);
    }

    auto getInterfaceInfoByFullName(StringRef fullName) -> InterfaceInfo::TypePtr
    {
        return fullNameInterfacesMap.lookup(fullName);
    }

    auto getGenericInterfaceInfoByFullName(StringRef fullName) -> GenericInterfaceInfo::TypePtr
    {
        return fullNameGenericInterfacesMap.lookup(fullName);
    }

  protected:

    mlir::Location loc(TextRange loc)
    {
        if (!loc)
        {
            return mlir::UnknownLoc::get(builder.getContext());
        }

        auto pos = loc->pos.textPos > 0 ? loc->pos.textPos : loc->pos.pos;
        //return loc1(sourceFile, fileName.str(), pos, loc->_end - pos);
        //return loc2(sourceFile, fileName.str(), pos, loc->_end - pos);
        return locFuseWithScope(
                    combine(
                        overwriteLoc,
                        loc2Fuse(sourceFile, mainSourceFileName.str(), pos, loc->_end - pos)));
    }

    mlir::Location loc1(ts::SourceFile sourceFile, std::string fileName, int start, int length)
    {
        auto fileId = getStringAttr(fileName);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start);
        auto begin = mlir::FileLineColLoc::get(builder.getContext(), 
            fileId, posLineChar.line + 1, posLineChar.character + 1);
        return begin;
    }

    mlir::Location loc2(ts::SourceFile sourceFile, std::string fileName, int start, int length)
    {
        auto fileId = getStringAttr(fileName);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start);
        auto begin = mlir::FileLineColLoc::get(builder.getContext(), fileId, 
            posLineChar.line + 1, posLineChar.character + 1);
        if (length <= 1)
        {
            return begin;
        }

        auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start + length - 1);
        auto end = mlir::FileLineColLoc::get(builder.getContext(), fileId, 
            endLineChar.line + 1, endLineChar.character + 1);
        //return mlir::FusedLoc::get(builder.getContext(), {begin, end});
        return begin;
    }

    mlir::Location loc2Fuse(ts::SourceFile sourceFile, std::string fileName, int start, int length)
    {
        auto fileId = getStringAttr(fileName);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start);
        auto begin = mlir::FileLineColLoc::get(builder.getContext(), fileId, 
            posLineChar.line + 1, posLineChar.character + 1);
        if (length <= 1)
        {
            return begin;
        }

        auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start + length - 1);
        auto end = mlir::FileLineColLoc::get(builder.getContext(), 
            fileId, endLineChar.line + 1, endLineChar.character + 1);
        //return mlir::FusedLoc::get(builder.getContext(), {begin, end});
        //return mlir::FusedLoc::get(builder.getContext(), {begin}, end);
        // TODO: why u did this way? because of loosing "column" info due to merging fused locations?
        //return mlir::FusedLoc::get(builder.getContext(), {begin});
        return begin;
    }

    mlir::Location locFuseWithScope(mlir::Location location)
    {
        if (!compileOptions.generateDebugInfo)
        {
            return location;
        }

        MLIRDebugInfoHelper mdi(builder, debugScope);
        //return mdi.combineWithCurrentLexicalBlockScope(location);
        return mdi.combineWithCurrentScope(location);
    }

    mlir::Location combine(mlir::Location parenLocation, mlir::Location location) 
    {
        if (isa<mlir::UnknownLoc>(parenLocation))
        {
            return location;
        }

        return mlir::FusedLoc::get(builder.getContext(), {parenLocation, location});  
    }

    mlir::Location stripMetadata(mlir::Location location)
    {
        MLIRDebugInfoHelper mdi(builder, debugScope);
        return mdi.stripMetadata(location);
    }    

    mlir::StringAttr getStringAttr(StringRef text)
    {
        return builder.getStringAttr(text);
    }

    mlir::Location loc_check(TextRange loc_)
    {
        assert(loc_->pos != loc_->_end);
        return loc(loc_);
    }

    mlir::LogicalResult parsePartialStatements(string src)
    {
        GenContext emptyContext{};
        return parsePartialStatements(src, emptyContext);
    }

    mlir::LogicalResult parsePartialStatements(string src, const GenContext& genContext, bool useRootNamesapce = true, bool file_d_ts = false)
    {
        Parser parser;
        // .d.ts will mark all variables as external (be careful)
        auto module = parser.parseSourceFile(file_d_ts ? S("partial.d.ts") : S("partial.ts"), src, ScriptTarget::Latest);

        MLIRNamespaceGuard nsGuard(currentNamespace);
        if (useRootNamesapce)
            currentNamespace = rootNamespace;

        DITableScopeT debugPartialCodeScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            if (!isa<mlir::UnknownLoc>(overwriteLoc))
            {
                overwriteLoc = stripMetadata(overwriteLoc);
            }
        }

        for (auto statement : module->statements)
        {
            if (mlir::failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    std::string to_print(mlir::Type type)
    {
        std::stringstream exportType;
        MLIRPrinter mp{};
        mp.printType<std::ostream>(exportType, type);
        return exportType.str();      
    }

    string to_wprint(mlir::Type type)
    {
        stringstream exportType;
        MLIRPrinter mp{};
        mp.printType<ostream>(exportType, type);
        return exportType.str();      
    }

    void printDebug(ts::Node node)
    {
        // Printer<llvm::raw_ostream> printer(llvm::dbgs());
        std::wcerr << std::endl << "dump ===============================================" << std::endl;
        Printer<std::wostream> printer(std::wcerr);
        printer.printNode(node);
        std::wcerr << std::endl << "end of dump ========================================" << std::endl;
    }

    // TODO: fix issue with cercular reference of include files
    std::pair<SourceFile, std::vector<SourceFile>> loadIncludeFile(mlir::Location location, StringRef fileName)
    {
        SmallString<256> fullPath;
        sys::path::append(fullPath, fileName);
        if (sys::path::extension(fullPath) == "")
        {
            fullPath += ".ts";
        }

        std::string ignored;
        auto id = sourceMgr.AddIncludeFile(std::string(fullPath), SMLoc(), ignored);
        if (!id)
        {
            emitError(location, "can't open file: ") << fullPath;
            return {SourceFile(), {}};
        }

        const auto *sourceBuf = sourceMgr.getMemoryBuffer(id);
        return loadSourceBuf(location, sourceBuf);
    }

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    llvm::SourceMgr &sourceMgr;

    SourceMgrDiagnosticHandlerEx sourceMgrHandler;

    MLIRTypeHelper mth;

    CompileOptions &compileOptions;

    /// A "module" matches a TypeScript source file: containing a list of functions.
    mlir::ModuleOp theModule;

    mlir::StringRef mainSourceFileName;

    mlir::StringRef path;

    /// An allocator used for alias names.
    llvm::BumpPtrAllocator stringAllocator;

    llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

    NamespaceInfo::TypePtr rootNamespace;

    NamespaceInfo::TypePtr currentNamespace;

    llvm::ScopedHashTable<StringRef, NamespaceInfo::TypePtr> fullNamespacesMap;

    llvm::ScopedHashTable<StringRef, GenericFunctionInfo::TypePtr> fullNameGenericFunctionsMap;

    llvm::ScopedHashTable<StringRef, EnumInfo::TypePtr> fullNameEnumsMap;

    llvm::ScopedHashTable<StringRef, ClassInfo::TypePtr> fullNameClassesMap;

    llvm::ScopedHashTable<StringRef, GenericClassInfo::TypePtr> fullNameGenericClassesMap;

    llvm::ScopedHashTable<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMap;

    llvm::ScopedHashTable<StringRef, GenericInterfaceInfo::TypePtr> fullNameGenericInterfacesMap;

    llvm::ScopedHashTable<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMap;

    llvm::ScopedHashTable<StringRef, mlir::LLVM::DIScopeAttr> debugScope;

    // helper to get line number
    Parser parser;
    ts::SourceFile sourceFile;

    bool declarationMode;

    std::stringstream declExports;
    mlir::SmallPtrSet<mlir::Type, 32> exportCheckedDependenciesTypes;
    mlir::SmallPtrSet<mlir::Type, 32> exportedTypes;

    Stages stage;

private:
    std::string label;
    mlir::Block* tempEntryBlock;
    mlir::ModuleOp tempModule;
    mlir_ts::FuncOp tempFuncOp;
    mlir::Location overwriteLoc;
};
} // namespace

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    auto showLineCharPos = false;

    Parser parser;
    auto sourceFile = parser.parseSourceFile(stows(static_cast<std::string>(fileName)),
                                             stows(static_cast<std::string>(source)), ScriptTarget::Latest);

    stringstream s;

    FuncT<> visitNode;
    ArrayFuncT<> visitArray;

    auto intent = 0;

    visitNode = [&](Node child) -> Node {
        for (auto i = 0; i < intent; i++)
        {
            s << "\t";
        }

        if (showLineCharPos)
        {
            auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->pos);
            auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->_end);

            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S("(")
              << posLineChar.line + 1 << S(":") << posLineChar.character + 1 << S(") - ") << child->_end << S("(")
              << endLineChar.line + 1 << S(":") << endLineChar.character << S(") ]") << std::endl;
        }
        else
        {
            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S(" - ")
              << child->_end << S(" ]") << std::endl;
        }

        intent++;
        ts::forEachChild(child, visitNode, visitArray);
        intent--;

        return undefined;
    };

    visitArray = [&](NodeArray<Node> array) -> Node {
        for (auto node : array)
        {
            visitNode(node);
        }

        return undefined;
    };

    auto result = forEachChild(sourceFile.as<Node>(), visitNode, visitArray);
    return convertWideToUTF8(s.str());
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromMainSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName,
                                        const llvm::SourceMgr &sourceMgr, CompileOptions &compileOptions)
{
    auto path = llvm::sys::path::parent_path(fileName);
    MLIRGenImpl mlirGenImpl(context, fileName, path, sourceMgr, compileOptions);
    auto [sourceFile, includeFiles] = mlirGenImpl.loadMainSourceFile();
    return mlirGenImpl.mlirGenSourceFile(sourceFile, includeFiles);
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromSource(const mlir::MLIRContext &context, SMLoc &smLoc, const llvm::StringRef &fileName,
                                        const llvm::SourceMgr &sourceMgr, CompileOptions &compileOptions)
{
    auto path = llvm::sys::path::parent_path(fileName);
    MLIRGenImpl mlirGenImpl(context, fileName, path, sourceMgr, compileOptions);
    auto [sourceFile, includeFiles] = mlirGenImpl.loadSourceFile(smLoc);
    return mlirGenImpl.mlirGenSourceFile(sourceFile, includeFiles);
}

} // namespace typescript
