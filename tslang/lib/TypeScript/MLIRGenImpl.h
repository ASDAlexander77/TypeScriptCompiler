// Private implementation header for the MLIRGen translation units.
// MLIRGenImpl is being split across several .cpp files (see docs/MLIRGen-refactoring-review.md SS1);
// method bodies still defined inline here are pending extraction.
// Include only from lib/TypeScript/MLIRGen*.cpp.
#pragma once

// TODO: it seems in Jit mode, LLVM Engine can resolve external references from loading DLLs

#ifdef GC_ENABLE
#define ADD_GC_ATTRIBUTE true
#endif

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DiagnosticHelper.h"

#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/MLIRLogic/MLIRGenContext.h"
#include "TypeScript/MLIRLogic/MLIRNamespaceGuard.h"
#include "TypeScript/MLIRLogic/MLIRLocationGuard.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRValueGuard.h"
#include "TypeScript/MLIRLogic/MLIRDebugInfoHelper.h"
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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#ifdef ENABLE_ASYNC
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
//#include "llvm/IR/DebugInfoMetadata.h"

#include "TypeScript/MLIRLogic/MLIRGenContextDefines.h"

#include <algorithm>
#include <iterator>
#include <type_traits>

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

// Defined in MLIRGen.cpp; also used by TypeScriptOps.cpp.
CompileOptions &getCompileOptions();
void setCompileOptions(CompileOptions &compileOptions);

namespace typescript
{
namespace mlirgen
{

enum class IsGeneric
{
    False,
    True,
    NoDefaults
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
            compileOptions,
            [this](StringRef name) { return getClassInfoByFullName(name); },
            [this](StringRef name) { return getGenericClassInfoByFullName(name); },
            [this](StringRef name) { return getInterfaceInfoByFullName(name); },
            [this](StringRef name) { return getGenericInterfaceInfoByFullName(name); }),
          compileOptions(compileOptions), 
          mainSourceFileName(fileNameParam),
          path(pathParam),
          declarationMode(false),
          tempEntryBlock(nullptr),
          overwriteLoc(mlir::UnknownLoc::get(builder.getContext()))
    {
        setCompileOptions(compileOptions);

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

    // RAII scope switching the current source file and file name (used for locations and debug info).
    class SourceFileScope
    {
      public:
        SourceFileScope(MLIRGenImpl &mlirGenImpl, ts::SourceFile newSourceFile, llvm::StringRef newFileName)
            : sourceFileGuard(mlirGenImpl.sourceFile, newSourceFile),
              fileNameGuard(mlirGenImpl.mainSourceFileName, newFileName)
        {
        }

        // interns the file name from the source file's wide file name
        SourceFileScope(MLIRGenImpl &mlirGenImpl, ts::SourceFile newSourceFile)
            : SourceFileScope(mlirGenImpl, newSourceFile,
                              llvm::StringRef(convertWideToUTF8(newSourceFile->fileName)).copy(mlirGenImpl.stringAllocator))
        {
        }

      private:
        MLIRValueGuard<ts::SourceFile> sourceFileGuard;
        MLIRValueGuard<llvm::StringRef> fileNameGuard;
    };

    mlir::LogicalResult report(SourceFile module, const std::vector<SourceFile> &includeFiles);

    std::pair<SourceFile, std::vector<SourceFile>> loadMainSourceFile();

    std::pair<SourceFile, std::vector<SourceFile>> loadSourceFile(SMLoc loc);

    std::pair<SourceFile, std::vector<SourceFile>> loadSourceBuf(mlir::Location location, const llvm::MemoryBuffer *sourceBuf, bool isMain = false);

    mlir::LogicalResult showMessages(SourceFile module, std::vector<SourceFile> includeFiles);

    mlir::ModuleOp mlirGenSourceFile(SourceFile module, std::vector<SourceFile> includeFiles);

  private:
    mlir::LogicalResult mlirGenCodeGenInit(SourceFile module);

#ifdef GENERATE_IMPORT_INFO_USING_D_TS_FILE
    /// Create a dependency declaration file for `--emit=dll` option.
    ///
    mlir::LogicalResult createDependencyDeclarationFile(StringRef outputFilename,
                                            StringRef dependencyDeclFileBody);
#endif    

    mlir::LogicalResult createDeclarationExportGlobalVar(const GenContext &genContext);
    mlir::LogicalResult createGenericClassDeclarationExportGlobalVar(const GenContext &genContext);

    bool isCodeStatment(SyntaxKind kind);

    int processStatements(NodeArray<Statement> statements,
                          const GenContext &genContext,
                          bool isRoot = false);

    bool hasGlobalCode(NodeArray<Statement> statements);

    // appends GlobalConstructorOp after the last one in the module; LAST priority so it runs after CRT init
    void addGlobalConstructor(mlir::Location location, StringRef funcName);

    mlir::LogicalResult generateGlobalEntryCode(mlir::Location location, NodeArray<Statement> statements,
                          const GenContext &genContext);

    mlir::LogicalResult outputDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages,
                                          int notResolved);

    mlir::LogicalResult mlirDiscoverAllDependencies(SourceFile module, std::vector<SourceFile> includeFiles = {});

    mlir::LogicalResult mlirCodeGenModule(SourceFile module, std::vector<SourceFile> includeFiles = {},
                                          bool validate = true, bool isMain = true);

    bool registerNamespace(llvm::StringRef namePtr, bool isFunctionNamespace = false);

    mlir::LogicalResult exitNamespace();

    mlir::LogicalResult mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext);

    mlir::LogicalResult mlirGenInclude(mlir::Location location, StringRef filePath, const GenContext &genContext);

    mlir::LogicalResult mlirGenImportSharedLib(mlir::Location location, StringRef filePath, bool dynamic, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ImportDeclaration importDeclarationAST, const GenContext &genContext);

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

    mlir::LogicalResult mlirGenBody(Node body, const GenContext &genContext);

    void clearState(NodeArray<Statement> statements);

    mlir::LogicalResult mlirGen(NodeArray<Statement> statements, const GenContext &genContext);

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

    mlir::LogicalResult mlirGen(ts::Block blockAST, const GenContext &genContext, int skipStatements = 0);

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
            // NOTE: upward mailbox into caller context (process-once drain) - see docs/MLIRGen-refactoring-review.md A7
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
                    LLVM_DEBUG(llvm::dbgs() << "\n!! failed: " << print(statement) << "\n";);

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
                // NOTE: upward mailbox into caller context (process-once) - see docs/MLIRGen-refactoring-review.md A7
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

    mlir::LogicalResult mlirGen(Statement statementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ExpressionStatement expressionStatementAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(Expression expressionAST, const GenContext &genContext);

    // inferType helpers; return true when the template kind matched and inference was handled

    bool tryInferNamedGeneric(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                              StringMap<mlir::Type> &results);

    bool tryInferClass(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext);

    bool tryInferInterface(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                           StringMap<mlir::Type> &results, const GenContext &genContext);

    bool tryInferArray(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext);

    // TODO: finish it
    template <typename T>
    bool tryInferTupleFields(mlir::Location location, mlir_ts::TupleType tempTuple, T typeTuple,
                             StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        for (auto tempFieldInfo : tempTuple.getFields())
        {
            auto index = typeTuple.getIndex(tempFieldInfo.id);
            if (index >= 0)
            {
                inferType(location, tempFieldInfo.type, typeTuple.getFieldInfo(index).type, results, genContext);
            }
            else
            {
                return true;
            }
        }

        return true;
    }

    bool tryInferTuple(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext);

    bool tryInferOptional(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                          StringMap<mlir::Type> &results, const GenContext &genContext);

    bool tryInferFunction(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                          StringMap<mlir::Type> &results, const GenContext &genContext);

    bool tryInferUnion(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext);

    void inferType(mlir::Location location, mlir::Type templateType, mlir::Type concreteType, StringMap<mlir::Type> &results, const GenContext &genContext);

    void inferTypeFuncType(mlir::Location location, mlir::ArrayRef<mlir::Type> tempfuncType, mlir::ArrayRef<mlir::Type> funcType,
                           StringMap<mlir::Type> &results, const GenContext &genContext);

    bool isGenericFunctionReference(mlir::Value functionRefValue);

    mlir::Type instantiateSpecializedFunctionTypeHelper(mlir::Location location, mlir::Value functionRefValue,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext);

    mlir::Type instantiateSpecializedFunctionTypeHelper(mlir::Location location, FunctionLikeDeclarationBase funcDecl,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext);

    void rollbackPostponedErrorMessages(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> *postponedMessages, size_t size);

    ValueOrLogicalResult instantiateSpecializedFunction(mlir::Location location,
        mlir::Value functionRefValue, mlir::Type recieverType, const GenContext &genContext);

    mlir::LogicalResult appendInferredTypes(mlir::Location location,
                                            llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                            StringMap<mlir::Type> &inferredTypes, IsGeneric &anyNamedGenericType,
                                            GenContext &genericTypeGenContext,
                                            bool arrayMerge = false, bool noExtendsTest = false);

    std::pair<mlir::LogicalResult, bool> resolveGenericParamFromFunctionCall(mlir::Location location, mlir::Type paramType, mlir::Value argOp, int paramIndex,
        GenericFunctionInfo::TypePtr functionGenericTypeInfo, IsGeneric &anyNamedGenericType,  GenContext &genericTypeGenContext);

    mlir::LogicalResult resolveGenericParamsFromFunctionCall(mlir::Location location,
                                                             GenericFunctionInfo::TypePtr functionGenericTypeInfo,
                                                             NodeArray<TypeNode> typeArguments,
                                                             bool skipThisParam,
                                                             IsGeneric &anyNamedGenericType,
                                                             GenContext &genericTypeGenContext);

    std::tuple<mlir::LogicalResult, mlir_ts::FunctionType, std::string> instantiateSpecializedFunction(
        mlir::Location location, StringRef name, NodeArray<TypeNode> typeArguments, bool skipThisParam, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext);

    std::pair<mlir::LogicalResult, FunctionPrototypeDOM::TypePtr> getFuncArgTypesOfGenericMethod(
        FunctionLikeDeclarationBase functionLikeDeclarationAST, ArrayRef<TypeParameterDOM::TypePtr> typeParams,
        bool discoverReturnType, const GenContext &genContext);

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               NodeArray<TypeNode> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics = false);

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               ArrayRef<mlir::Type> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics = false);

    std::pair<mlir::LogicalResult, mlir::Type> instantiateSpecializedInterfaceType(
        mlir::Location location, mlir_ts::InterfaceType genericInterfaceType, NodeArray<TypeNode> typeArguments,
        const GenContext &genContext, bool allowNamedGenerics = false);

    ValueOrLogicalResult mlirGenSpecialized(mlir::Location location, mlir::Value genResult,
                                            NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext);

    ValueOrLogicalResult mlirGen(Expression expression, NodeArray<TypeNode> typeArguments, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ExpressionWithTypeArguments expressionWithTypeArgumentsAST,
                                 const GenContext &genContext);

    ValueOrLogicalResult registerVariableInThisContext(mlir::Location location, StringRef name, mlir::Type type,
                                                       const GenContext &genContext);

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
        VariableDeclarationInfo(CompileOptions& compileOptions) : compileOptions(compileOptions), variableName(), fullName(), initial(), type(), storage(), globalOp(), varClass(),
            scope{VariableScope::Local}, isFullName{false}, isGlobal{false}, isConst{false}, isExternal{false}, isExport{false}, isImport{false}, 
            isSpecialization{false}, allocateOutsideOfOperation{false}, allocateInContextThis{false}, comdat{Select::NotSet}, deleted{false}, isUsed{false},
            needsIdentityStorage{false}, typeAndInitResolved{false}
        {
        };

        VariableDeclarationInfo(
            CompileOptions& compileOptions,
            TypeValueInitFuncType func_, 
            std::function<StringRef(StringRef)> getFullNamespaceName_) : VariableDeclarationInfo(compileOptions) 
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

        void detectFlags(bool isFullName_, VariableClass varClass_, bool forceLocalVar, const GenContext &genContext)
        {
            varClass = varClass_;
            isFullName = isFullName_;
            
            if (isFullName_ || !genContext.funcOp)
            {
                scope = VariableScope::Global;
            }

            if (forceLocalVar)
            {
                scope = VariableScope::Local;
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

            // a const binding whose type is a value-typed aggregate with a bound-method
            // field (e.g. a generator wrapper's `next`) has mutable state but no
            // pointer-like representation -- it still needs real storage so that
            // mutation through the bound method is visible across property accesses.
            // See docs/const-let-storage-design.md.
            MLIRTypeHelper mth(builder.getContext(), compileOptions);
            needsIdentityStorage = mth.hasBoundMethodField(type);
            if (needsIdentityStorage)
            {
                return mlir::success();
            }

            if (varClass == VariableType::ConstRef)
            {
                MLIRCodeLogic mcl(builder, compileOptions);
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
            // `func` re-runs mlirGen on the initializer expression, which emits real
            // ops (e.g. re-invokes a call expression) -- must not run twice. A const
            // binding that turns out to need identity storage calls this once via
            // processConstRef() and then again via createLocalVariable(); guard so
            // the second call just reuses what's already resolved.
            if (typeAndInitResolved)
            {
                return type ? mlir::success() : mlir::failure();
            }

            typeAndInitResolved = true;

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
            // readWriteAccess really means "this binding has real RefType storage that
            // must be loaded through / can be captured by reference" -- it is not a
            // const-reassignment check (this codebase doesn't enforce one at this
            // layer). A const with identity storage (see needsIdentityStorage /
            // hasBoundMethodField) has real storage exactly like `let`, so it needs
            // the same load-through-ref and by-ref-capture treatment.
            if (!isConst || varClass == VariableType::ConstRef || needsIdentityStorage)
            {
                varDecl->setReadWriteAccess();
                // TODO: HACK: to mark var as local and ignore when capturing
                if (varClass == VariableType::ConstRef)
                {
                    varDecl->setIgnoreCapturing();
                }
            }

            varDecl->setUsing(varClass.isUsing);

            if (varClass.atomic)
            {
                varDecl->setAtomic(varClass.ordering, varClass.syncscope);
            }

            varDecl->setVolatile(varClass.isVolatile);
            varDecl->setNonTemporal(varClass.nonTemporal);
            varDecl->setInvariant(varClass.invariant);

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

        CompileOptions& compileOptions;

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
        bool needsIdentityStorage;
        bool typeAndInitResolved;
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
            auto storeOp = builder.create<mlir_ts::StoreOp>(location, variableDeclarationInfo.initial, variableDeclarationInfo.storage);
            if (variableDeclarationInfo.varClass.atomic)
            {
                storeOp->setAttr(ATOMIC_ATTR_NAME, builder.getBoolAttr(true));
                storeOp->setAttr(ORDERING_ATTR_NAME, builder.getI32IntegerAttr(variableDeclarationInfo.varClass.ordering));
                storeOp->setAttr(SYNCSCOPE_ATTR_NAME, builder.getStringAttr(variableDeclarationInfo.varClass.syncscope));
            }

            if (variableDeclarationInfo.varClass.isVolatile)
            {
                storeOp->setAttr(VOLATILE_ATTR_NAME, builder.getBoolAttr(true));
            }            

            if (variableDeclarationInfo.varClass.nonTemporal)
            {
                storeOp->setAttr(NONTEMPORAL_ATTR_NAME, builder.getBoolAttr(true));
            }            

            // if (variableDeclarationInfo.varClass.invariant)
            // {
            //     storeOp->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
            // }
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

        genContextWithNameReceiver.isExportVarReceiver = variableDeclarationInfo.isExport;

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
                    if (mlir::failed(createGlobalVariableInitialization(location, globalOp, variableDeclarationInfo, genContext)))
                    {
                        return mlir::failure();
                    }
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
            auto storeOp = builder.create<mlir_ts::StoreOp>(location, variableDeclarationInfo.initial, address);
            if (variableDeclarationInfo.varClass.atomic)
            {
                storeOp->setAttr(ATOMIC_ATTR_NAME, builder.getBoolAttr(true));
                storeOp->setAttr(ORDERING_ATTR_NAME, builder.getI32IntegerAttr(variableDeclarationInfo.varClass.ordering));
                storeOp->setAttr(SYNCSCOPE_ATTR_NAME, builder.getStringAttr(variableDeclarationInfo.varClass.syncscope));
            }

            if (variableDeclarationInfo.varClass.isVolatile)
            {
                storeOp->setAttr(VOLATILE_ATTR_NAME, builder.getBoolAttr(true));
            }               

            if (variableDeclarationInfo.varClass.nonTemporal)
            {
                storeOp->setAttr(NONTEMPORAL_ATTR_NAME, builder.getBoolAttr(true));
            }            

            // if (variableDeclarationInfo.varClass.invariant)
            // {
            //     storeOp->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
            // }
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

    mlir::LogicalResult registerVariableDeclaration(mlir::Location location, VariableDeclarationDOM::TypePtr variableDeclaration, struct VariableDeclarationInfo &variableDeclarationInfo, bool showWarnings, const GenContext &genContext);

    mlir::Type registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                                TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false, bool forceLocalVar = false);

    // TODO: to support '...' u need to use 'processOperandSpreadElement' and instead of "index" param use "next" logic
    ValueOrLogicalResult processDeclarationArrayBindingPatternSubPath(
        mlir::Location location, int index, mlir::Type type, mlir::Value init, 
        bool isDotDotDot, bool isIterator, bool isArrayLike, mlir::Value arrayLikeLengthValue, mlir::Type arrayLikeElementType, const GenContext &genContext);

    mlir::LogicalResult processDeclarationArrayBindingPattern(mlir::Location location, ArrayBindingPattern arrayBindingPattern,
                                               VariableClass varClass,
                                               TypeValueInitFuncType func,
                                               const GenContext &genContext);

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
        mlir::Location location, BindingElement objectBindingElement, mlir::Type type, mlir::Value init, const GenContext &genContext);

    ValueOrLogicalResult processDeclarationObjectBindingPatternSubPathSpread(
        mlir::Location location, ObjectBindingPattern objectBindingPattern, mlir::Type type, mlir::Value init, const GenContext &genContext);

    mlir::LogicalResult processDeclarationObjectBindingPattern(mlir::Location location, ObjectBindingPattern objectBindingPattern,
                                                VariableClass varClass,
                                                TypeValueInitFuncType func,
                                                const GenContext &genContext);

    mlir::LogicalResult processDeclarationName(DeclarationName name, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false);

    mlir::LogicalResult processDeclaration(NamedDeclaration item, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings = false);

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
            
            // in case we have receiver but next function is not arrow declaration, we need to remove receiver to stop cofusion with next nested level
            // so if arrow is part of call, it will be considered as receiver of initialization which is wrong,
            // example: const seq = f( (x) => x + 1 ); 
            // seq will become name of function
            if (initializer != SyntaxKind::ArrowFunction) 
            {
                genContextWithTypeReceiver.receiverName = StringRef();
                genContextWithTypeReceiver.isGlobalVarReceiver = false;
            }

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

    mlir::LogicalResult mlirGen(VariableDeclaration item, VariableClass varClass, const GenContext &genContext);

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

    mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext);

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

        fieldInfos.push_back({fieldId, fieldType, false, mlir_ts::AccessLevel::Public});

        return mlir::success();
    }

    mlir::Type mlirGenParameterObjectOrArrayBinding(Node name, const GenContext &genContext);

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
        SignatureDeclarationBase parametersContextAST, const GenContext &genContext);

    std::tuple<std::string, std::string> getNameOfFunction(SignatureDeclarationBase signatureDeclarationBaseAST,
                                                           const GenContext &genContext);

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
            auto cachedFuncType = funcIt->second.funcType;
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

    void processFunctionAttributes(SmallVector<mlir::NamedAttribute> &attrs, const GenContext &genContext)
    {
        if (genContext.specialization)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "specialization"), mlir::UnitAttr::get(builder.getContext())});
        }
    }

    bool processFunctionAttributes(mlir::Location location, StringRef fullName,
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, SmallVector<mlir::NamedAttribute> &attrs, const GenContext &genContext,
        bool suppressExport = false)
    {
#ifdef ADD_GC_ATTRIBUTE
        attrs.push_back({builder.getIdentifier(TS_GC_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
#endif
        // add decorations, "noinline, optnone"

        iterateDecorators(functionLikeDeclarationBaseAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
            if (isFuncAttr(name))
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), name), mlir::UnitAttr::get(builder.getContext())});
            }

            if (name == "varargs") 
            {
                attrs.push_back({mlir::StringAttr::get(builder.getContext(), "func.varargs"), mlir::BoolAttr::get(builder.getContext(), true)});
            }

            if (name == "used") {
                builder.create<mlir_ts::AppendToUsedOp>(location, fullName);
            }
        });

        // add modifiers
        auto dllExport = !suppressExport
            && (getExportModifier(functionLikeDeclarationBaseAST)
                || ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllExport) == InternalFlags::DllExport));
        if (dllExport)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "export"), mlir::UnitAttr::get(builder.getContext())});
        }

        auto dllImport = ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllImport) == InternalFlags::DllImport);
        if (dllImport)
        {
            attrs.push_back({mlir::StringAttr::get(builder.getContext(), "import"), mlir::UnitAttr::get(builder.getContext())});
        }

        processFunctionAttributes(attrs, genContext);

        return dllExport;
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, mlir::LogicalResult, bool> mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext);

    void resetScope() {
        // we need to remove "this" reference when we generate generic class inside other function of class
        symbolTable.insert("this", {mlir::Value(), {}});
        //symbolTable.insert(THIS_ALIAS, {mlir::Value(), {}});
    }

    mlir::LogicalResult discoverFunctionReturnTypeAndCapturedVars(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, StringRef name, SmallVector<mlir::Type> &argTypes,
        const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext);

    mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext);

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> mlirGenFunctionGenerator(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext);

    // Builds the synthetic non-generator declaration (method/function whose body just returns the
    // generator wrapper object literal with its `next` method) that mlirGenFunctionGenerator generates
    // from. Factored out so callers that only need the correctly-typed prototype (e.g. object-literal
    // method prototype registration) can reuse it without running full body codegen twice.
    FunctionLikeDeclarationBase buildGeneratorWrapperDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir::Location location);

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
            // already registered - but the registration itself typically happens during
            // Stages::Discovering (before addGenericFunctionDeclarationToExport's
            // isAddedToExport gate, which only actually emits once stage ==
            // Stages::SourceGeneration, will do anything) - retry the export step alone
            // using the existing GenericFunctionInfo rather than skipping it entirely.
            // Mirrors registerGenericClass's identical gotcha-3 fix.
            if (getExportModifier(functionLikeDeclarationBaseAST))
            {
                addGenericFunctionDeclarationToExport(lookupGenericFunctionMap(name));
            }

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
        newGenericFunctionPtr->thisType = genContext.thisType;
        newGenericFunctionPtr->thisClassType = genContext.thisClassType;
        newGenericFunctionPtr->sourceFile = sourceFile;
        newGenericFunctionPtr->fileName = mainSourceFileName;

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

        // support dynamic loading: a generic function is never instantiated in this
        // module if nothing here uses it concretely, so mlirGenFunctionPrototype's own
        // addFunctionDeclarationToExport call never runs for the bare template (it only
        // fires for a SPECIALIZED instantiation) - the bare template needs to be
        // exported here instead, the one place every generic function declaration
        // passes through regardless of whether it is ever instantiated locally. Mirrors
        // registerGenericClass.
        if (getExportModifier(functionLikeDeclarationBaseAST))
        {
            addGenericFunctionDeclarationToExport(newGenericFunctionPtr);
        }

        return {mlir::success(), name};
    }

    static FunctionEntry makeFunctionEntry(mlir_ts::FuncOp funcOp)
    {
        return FunctionEntry{funcOp.getName().str(), mlir::cast<mlir_ts::FunctionType>(funcOp.getFunctionType())};
    }

    bool registerFunctionOp(FunctionPrototypeDOM::TypePtr funcProto, mlir_ts::FuncOp funcOp);

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> mlirGenFunctionLikeDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext);

    mlir::LogicalResult mlirGenStaticFieldDeclarationDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, 
        StringRef name, mlir::Type type, mlir_ts::AccessLevel accessLevel, const GenContext &genContext)
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

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, accessLevel);

        return mlir::success();
    }    

    mlir::LogicalResult mlirGenFunctionLikeDeclarationDynamicImport(mlir::Location location, StringRef funcName, 
        mlir_ts::FunctionType functionType, StringRef dllFuncName, const GenContext &genContext, bool isFullNamespaceName = true);

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto,
                                             const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, mlir::Type retType, const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionExit(mlir::Location location, const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionCapturedParam(mlir::Location location, int &firstIndex,
                                                     FunctionPrototypeDOM::TypePtr funcProto,
                                                     mlir::Block::BlockArgListType arguments,
                                                     const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionCapturedParamIfObject(mlir::Location location, int &firstIndex,
                                                             FunctionPrototypeDOM::TypePtr funcProto,
                                                             mlir::Block::BlockArgListType arguments,
                                                             const GenContext &genContext);

    ValueOrLogicalResult optionalValueOrUndefinedExpression(mlir::Location location, mlir::Value condValue, Expression expression, const GenContext &genContext)
    {
        return optionalValueOrUndefined(location, condValue, [&](auto genContext) { return mlirGen(expression, genContext); }, genContext);
    }

    ValueOrLogicalResult optionalValueOrUndefined(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult(const GenContext &)> exprFunc, const GenContext &genContext)
    {
        return conditionalValue(location, condValue, 
            [&]() { 
                auto result = exprFunc(genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto value = V(result);
                auto optValue = 
                    isa<mlir_ts::OptionalType>(value.getType())
                        ? value
                        : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                return ValueOrLogicalResult(optValue); 
            }, 
            [&](mlir::Type trueValueType) { 
                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, trueValueType);
                return ValueOrLogicalResult(optUndefValue); 
            });
    }

    ValueOrLogicalResult anyOrUndefined(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult(const GenContext &)> exprFunc, const GenContext &genContext)
    {
        return conditionalValue(location, condValue, 
            [&]() { 
                auto result = exprFunc(genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto value = V(result);
                auto anyValue = V(builder.create<mlir_ts::CastOp>(location, getAnyType(), value));
                return ValueOrLogicalResult(anyValue); 
            }, 
            [&](mlir::Type trueValueType) {
                auto undefValue = builder.create<mlir_ts::UndefOp>(location, getUndefinedType());
                auto anyUndefValue = V(builder.create<mlir_ts::CastOp>(location, trueValueType, undefValue));
                return ValueOrLogicalResult(anyUndefValue); 
            });
    }

    ValueOrLogicalResult conditionalValue(mlir::Location location, mlir::Value condValue, 
        std::function<ValueOrLogicalResult()> trueValue, 
        std::function<ValueOrLogicalResult(mlir::Type trueValueType)> falseValue)
    {
        MLIRCodeLogicHelper mclh(builder, location, compileOptions);
        return mclh.conditionalValue(condValue, trueValue, falseValue);
    }    

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
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext);
    mlir::LogicalResult mlirGenFunctionParamsBindings(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                      mlir::Block::BlockArgListType arguments,
                                                      const GenContext &genContext);


    mlir::LogicalResult mlirGenFunctionParams(mlir::Location location, int firstIndex, mlir::Block::BlockArgListType arguments, const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionCaptures(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                            StringRef name, mlir_ts::FuncOp funcOp, FunctionPrototypeDOM::TypePtr funcProto,
                                            const GenContext &genContext);

    mlir::LogicalResult mlirGenFunctionBody(mlir::Location location, StringRef funcName, StringRef fullFuncName,
                                            mlir_ts::FunctionType funcType, std::function<mlir::LogicalResult(mlir::Location, const GenContext &)> funcBody,                                            
                                            const GenContext &genContext,
                                            int firstParam = 0, bool isPublic = false);

    ValueOrLogicalResult mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(AsExpression asExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ComputedPropertyName computedPropertyNameAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext);

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

    ValueOrLogicalResult mlirGen(YieldExpression yieldExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(AwaitExpression awaitExpressionAST, const GenContext &genContext);

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
        mlir_ts::FuncOp funcOp = genContext.funcOp;
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

            emitError(location) << "can't find return variable, seems your function type has 'void' return type.";
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

    mlir::LogicalResult addSafeCastStatement(Expression exprIn, mlir::Type safeType, bool inverse, ElseSafeCase* elseSafeCase, const GenContext &genContext)
    {
        auto expr = stripParenthesesAndUntangleEquals(exprIn);

        auto isNotLocalVariable = false;
        auto location = loc(expr);
        auto nameStr = MLIRHelper::getName(expr.as<Node>(), stringAllocator);
        auto result = mlirGen(expr, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result);
        auto exprValue = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!! Is Safe Type the same: [" << exprValue.getType() << "] and [" << safeType << "]\n");

        if (isSafeTypeTheSameAndNoNeedToCast(exprValue.getType(), safeType))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! == Yes\n");
            return mlir::success();
        }

        if (nameStr.empty())
        {
            isNotLocalVariable = true;
            nameStr = ".safe_cast";
            if (expr == SyntaxKind::PropertyAccessExpression) 
            {
                nameStr = mlir::StringRef(print(expr)).copy(stringAllocator);
            }
        }

        if (elseSafeCase)
        {
            elseSafeCase->expr = expr;
        }

        auto result2 = addSafeCastStatement(location, nameStr, exprValue, safeType, inverse, elseSafeCase, genContext);

        // we need to register pair type+field to associate to variable
        if (isNotLocalVariable)
        {
            if (auto safeValue = resolveIdentifier(location, nameStr, genContext))
            {
                if (auto safeValueOp = safeValue.getDefiningOp<mlir_ts::SafeCastOp>())
                {
                    if (expr == SyntaxKind::PropertyAccessExpression) 
                    {
                        auto propAccess = expr.as<PropertyAccessExpression>();
                        auto objType = evaluate(propAccess->expression, genContext);
                        LLVM_DEBUG(llvm::dbgs() << "\n!! Safe Type map for: " << nameStr << " of " << objType << " is [" << safeValue.getType() << "]\n");
                        safeTypesMap.insert({ objType, nameStr }, safeValue);
                    }
                }
            }
        }

        return result2;
    }    

    mlir::LogicalResult addSafeCastStatement(mlir::Location location, StringRef parameterName, mlir::Value exprValue, mlir::Type safeType, bool inverse, ElseSafeCase* elseSafeCase, const GenContext &genContext)
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

        // we need to create dummy op to be able to use both values with cast and without cast
        auto wrappedValue = builder.create<mlir_ts::SafeCastOp>(location, castedValue.getType(), castedValue, exprValue);

        return 
            !!registerVariable(
                location, parameterName, false, VariableType::Const,
                [&](mlir::Location, const GenContext &) -> TypeValueInitType
                {
                    return {safeType, wrappedValue, TypeProvided::Yes};
                },
                genContext, false, true) ? mlir::success() : mlir::failure();        
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
                else if (text == S("null"))
                {
                    typeToken = nf.createToken(SyntaxKind::NullKeyword);
                }                
                else if (text == S("undefined"))
                {
                    typeToken = nf.createToken(SyntaxKind::UndefinedKeyword);
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

    mlir::LogicalResult checkSafeCastNull(Expression val, Expression nullVal, bool inverse, ElseSafeCase *elseSafeCase, const GenContext &genContext)
    {
        auto expr = stripParentheses(nullVal);
        if (expr == SyntaxKind::NullKeyword)
        {
            return addSafeCastStatement(val, getNullType(), inverse, elseSafeCase, genContext);
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

    Expression stripParenthesesAndUntangleEquals(Expression exprVal)
    {
        auto expr = exprVal;
        while (expr == SyntaxKind::ParenthesizedExpression || expr == SyntaxKind::BinaryExpression)
        {
            if (expr == SyntaxKind::ParenthesizedExpression)
            {
                expr = expr.as<ParenthesizedExpression>()->expression;
                continue;
            }

            if (expr == SyntaxKind::BinaryExpression)
            {
                auto binExpr = expr.as<BinaryExpression>();
                auto op = (SyntaxKind)binExpr->operatorToken;
                if (op == SyntaxKind::EqualsToken)
                {
                    expr = binExpr->left;
                }
                else if (op == SyntaxKind::CommaToken)
                {
                    expr = binExpr->right;
                }
            }
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

                MLIRCodeLogic mcl(builder, compileOptions);
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
                                // NOTE: upward mailbox: alias must stay visible for following statements - see A7
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
                                    // NOTE: upward mailbox: alias must stay visible for following statements - see A7
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

        if (expr == SyntaxKind::BinaryExpression)
        {
            auto binExpr = expr.as<BinaryExpression>();
            auto op = (SyntaxKind)binExpr->operatorToken;
            if (op == SyntaxKind::AmpersandAmpersandToken)
            {
                auto left = binExpr->left;
                auto leftResult = checkSafeCast(left, conditionValue, elseSafeCase, genContext);
                if (mlir::failed(leftResult))
                {
                    return leftResult;
                }

                auto right = binExpr->right;                
                auto rightResult = checkSafeCast(right, conditionValue, elseSafeCase, genContext);
                if (mlir::failed(rightResult))
                {
                    return rightResult;
                }

                return mlir::success();
            }
        }

        return checkSafeCastOne(exprIn, conditionValue, elseSafeCase, genContext);
    }

    mlir::LogicalResult checkSafeCastOne(Expression exprIn, mlir::Value conditionValue, ElseSafeCase *elseSafeCase, const GenContext &genContext)
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

            // TODO: check if we need to do samething for SafeCastOp
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
                                    if (mlir::failed(checkSafeCastUndefined(right, left, !inverse, elseSafeCase, genContext)))
                                    {
                                        // null case
                                        if (mlir::failed(checkSafeCastNull(left, right, inverse, elseSafeCase, genContext)))
                                        {
                                            return checkSafeCastNull(right, left, inverse, elseSafeCase, genContext);
                                        }
                                    }
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

    mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(DoStatement doStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(WhileStatement whileStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ForStatement forStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ForInStatement forInStatementAST, const GenContext &genContext);

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

    mlir::LogicalResult mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(DebuggerStatement debuggerStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(BreakStatement breakStatementAST, const GenContext &genContext);

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
                // NOTE: upward mailbox into caller context (process-once drain) - see A7
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

    mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(TryStatement tryStatementAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext);

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

    ValueOrLogicalResult mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext);

    // TODO: rewrite code, you can set IfOp result type later, see function anyOrUndefined
    ValueOrLogicalResult mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext);

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
                SafeTypesMapScopeT safeTypesMapScope(safeTypesMap);
                checkSafeCast(leftExpression, V(result), &elseSafeCase, genContext);

                auto result = mlirGen(rightExpression, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                resultTrue = V(result);

                if (mlir::failed(instantiateGenericsForBinaryOp(location, leftExpressionValue, resultTrue, genContext))) 
                {
                    return mlir::failure();
                }
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

                if (mlir::failed(instantiateGenericsForBinaryOp(location, leftExpressionValue, resultFalse, genContext))) 
                {
                    return mlir::failure();
                }                
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

        if (mlir::failed(instantiateGenericsForBinaryOp(location, leftExpressionValue, resultTrue, genContext))) 
        {
            return mlir::failure();
        }        

        // sync left part
        if (resultType != resultTrue.getType())
        {
            CAST(resultTrue, location, resultType, resultTrue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto resultFalse = leftExpressionValue;

        if (mlir::failed(instantiateGenericsForBinaryOp(location, leftExpressionValue, resultFalse, genContext))) 
        {
            return mlir::failure();
        }

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

        // the length-based numeric-index rewrite below only makes sense when the left
        // side is actually a number (e.g. `i in arr`); a string-literal left side (e.g.
        // `"length" in arr` or `"push" in arr`) must fall through to the general
        // field-lookup path further down instead, otherwise we'd cast a string to an
        // index/int type and generate invalid IR.
        auto leftIsStringLiteral = binaryExpressionAST->left == SyntaxKind::StringLiteral
            || binaryExpressionAST->left == SyntaxKind::NoSubstitutionTemplateLiteral;

        if (!leftIsStringLiteral && evaluateProperty(binaryExpressionAST->right, LENGTH_FIELD_NAME, genContext))
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
                                               const GenContext &genContext);

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

                MLIRCodeLogicHelper mclh(builder, location, compileOptions);
                auto returnValue = mclh.conditionalValue(
                    cmpResult,
                    [&]() {
                        // TODO: test cast value
                        auto thisPtrValue = cast(location, getOpaqueType(), resultLeftValue, genContext);
                        return mlirGenInstanceOfOpaque(location, thisPtrValue, resultRightValue, genContext);
                    },
                    [&](mlir::Type trueType) { // default false value
                                                                             // compare typeOfValue
                        return ValueOrLogicalResult(builder.create<mlir_ts::ConstantOp>(location, getBooleanType(),
                                                                   builder.getBoolAttr(false)));
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
                result = leftInt << rightInt.urem(leftInt.getBitWidth());
                break;
            case SyntaxKind::GreaterThanGreaterThanToken:
                result = leftInt.ashr(rightInt.urem(leftInt.getBitWidth()));
                break;
            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                result = leftInt.lshr(rightInt.urem(leftInt.getBitWidth()));
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
            // JS bitwise/shift operators coerce both operands to Int32, so the shift
            // amount is masked mod 32 here regardless of the 64-bit APSInt width used
            // above to stage the float->int conversion.
            case SyntaxKind::LessThanLessThanToken:
                resultAPInt = leftAPInt.shl(rightAPInt.urem(32));
                break;
            case SyntaxKind::GreaterThanGreaterThanToken:
                resultAPInt = leftAPInt.ashr(rightAPInt.urem(32));
                break;
            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                resultAPInt = leftAPInt.lshr(rightAPInt.urem(32));
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

    void cloneAtomicAttributes(mlir::Operation* opSrc, mlir::Operation* opDest)
    {
        // copy attrs over
        if (auto atomicAttr = opSrc->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
        {
            auto orderingAttr = opSrc->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
            auto syncScopeAttr = opSrc->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
            opDest->setAttr(ATOMIC_ATTR_NAME, atomicAttr);
            opDest->setAttr(ORDERING_ATTR_NAME, orderingAttr);
            opDest->setAttr(SYNCSCOPE_ATTR_NAME, syncScopeAttr);
        }

        if (auto volatileAttr = opSrc->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
        {
            opDest->setAttr(VOLATILE_ATTR_NAME, volatileAttr);
        }      

        if (auto nonTemporalAttr = opSrc->getAttrOfType<mlir::BoolAttr>(NONTEMPORAL_ATTR_NAME))
        {
            opDest->setAttr(NONTEMPORAL_ATTR_NAME, nonTemporalAttr);
        }            

        // if (auto invariantAttr = opSrc->getAttrOfType<mlir::BoolAttr>(INVARIANT_ATTR_NAME))
        // {
        //     opDest->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
        // }
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

        // TODO: logic to support safe cast
        if (auto safeCastOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::SafeCastOp>())
        {
            leftExpressionValueBeforeCast = safeCastOp.getValue();
        }

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
            auto storeOp = builder.create<mlir_ts::StoreOp>(location, savingValue, loadOp.getReference());
            cloneAtomicAttributes(loadOp, storeOp);
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
            MLIRCodeLogic mcl(builder, compileOptions);
            auto propRef = mcl.GetReferenceFromValue(location, leftExpressionValueBeforeCast);
            if (!propRef)
            {
                emitError(location, "saving to constant object");
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
            MLIRCodeLogic mcl(builder, compileOptions);
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
            MLIRCodeLogic mcl(builder, compileOptions);
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

                MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, rightExpressionValue, builder.getI32IntegerAttr(index));
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
    
    ValueOrLogicalResult syncUnionTypes(mlir::Location location, mlir::Value &leftExpressionValue, mlir::Value &rightExpressionValue, const GenContext &genContext) 
    {
        auto isLeftUnion = false;
        if (auto leftUnionType = dyn_cast<mlir_ts::UnionType>(leftExpressionValue.getType()))
        {
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(location, leftUnionType, baseType))
            {
                isLeftUnion = true;
            }
        }

        auto isRightUnion = false;
        if (auto rightUnionType = dyn_cast<mlir_ts::UnionType>(rightExpressionValue.getType()))
        {
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(location, rightUnionType, baseType))
            {
                isRightUnion = true;
            }
        }

        if (isLeftUnion && isRightUnion)
        {
            // TODO: finish cast between unions
            emitError(location, "Binary Operation") << "can't be applied to different union types. Apply type cast before usage";            
            return mlir::failure();
        }

        if (isLeftUnion)
        {
            CAST(leftExpressionValue, location, rightExpressionValue.getType(), leftExpressionValue, genContext);
            return leftExpressionValue;
        }

        if (isRightUnion)
        {
            CAST(rightExpressionValue, location, leftExpressionValue.getType(), rightExpressionValue, genContext);
            return rightExpressionValue;
        }

        return mlir::success();
    }

    bool syncTypes(mlir::Location location, mlir::Type type, mlir::Value &leftExpressionValue, mlir::Value &rightExpressionValue, const GenContext &genContext)
    {
        auto hasType = leftExpressionValue.getType() == type ||
                            rightExpressionValue.getType() == type;
        if (hasType)
        {
            if (leftExpressionValue.getType() != type)
            {
                if (MLIRTypeCore::canHaveToPrimitiveMethod(leftExpressionValue.getType()))
                {
                    CAST(leftExpressionValue, location, getNumberType(), leftExpressionValue, genContext);
                }

                CAST(leftExpressionValue, location, type, leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != type)
            {
                if (MLIRTypeCore::canHaveToPrimitiveMethod(leftExpressionValue.getType()))
                {
                    CAST(rightExpressionValue, location, getNumberType(), rightExpressionValue, genContext);
                }

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

    // JS/TS `===`/`!==` never coerce: operands of clearly different primitive kinds
    // (boolean vs number, boolean vs string, string vs number) are simply unequal,
    // full stop. Everything downstream of this function (adjustTypesForBinaryOp's
    // numeric-sync loop, LogicalBinaryOpLowering) is shared with `==`/`!=`, which DO
    // coerce, so without this check `1 === true` was being widened to a common
    // numeric type just like `1 == true` and wrongly evaluated to `true`.
    bool isDefinitelyMismatchedForStrictEquals(mlir::Type leftType, mlir::Type rightType)
    {
        auto isBoolean = [](mlir::Type type) { return isa<mlir_ts::BooleanType>(type); };
        auto isString = [](mlir::Type type) { return isa<mlir_ts::StringType>(type); };
        auto isNumeric = [](mlir::Type type) { return type.isIntOrIndexOrFloat() && !isa<mlir_ts::BooleanType>(type); };

        auto leftIsBoolean = isBoolean(leftType);
        auto rightIsBoolean = isBoolean(rightType);
        auto leftIsString = isString(leftType);
        auto rightIsString = isString(rightType);
        auto leftIsNumeric = isNumeric(leftType);
        auto rightIsNumeric = isNumeric(rightType);

        // only fire when BOTH sides are known, unambiguous primitives (not any/union/
        // object/etc., which may still need the general coercion/toPrimitive machinery)
        auto leftIsKnownPrimitive = leftIsBoolean || leftIsString || leftIsNumeric;
        auto rightIsKnownPrimitive = rightIsBoolean || rightIsString || rightIsNumeric;
        if (!leftIsKnownPrimitive || !rightIsKnownPrimitive)
        {
            return false;
        }

        return (leftIsBoolean != rightIsBoolean) || (leftIsString != rightIsString) || (leftIsNumeric != rightIsNumeric);
    }

    // TODO: review it, seems like big hack
    mlir::LogicalResult adjustTypesForBinaryOp(mlir::Location location, SyntaxKind opCode, mlir::Value &leftExpressionValue,
                                               mlir::Value &rightExpressionValue, const GenContext &genContext)
    {
        if (opCode == SyntaxKind::CommaToken)
        {
            return mlir::success();
        }

        if ((opCode == SyntaxKind::EqualsEqualsEqualsToken || opCode == SyntaxKind::ExclamationEqualsEqualsToken)
            && isDefinitelyMismatchedForStrictEquals(leftExpressionValue.getType(), rightExpressionValue.getType()))
        {
            auto result = opCode == SyntaxKind::ExclamationEqualsEqualsToken;
            leftExpressionValue = builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(result));
            rightExpressionValue = builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(true));
            return mlir::success();
        }

        if (MLIRTypeCore::canHaveToPrimitiveMethod(leftExpressionValue.getType())
            && evaluateProperty(location, leftExpressionValue, SYMBOL_TO_PRIMITIVE, genContext)
            && !isa<mlir_ts::UndefinedType>(rightExpressionValue.getType())
            && !isa<mlir_ts::NullType>(rightExpressionValue.getType()))
        {
            auto type = isa<mlir_ts::StringType>(rightExpressionValue.getType()) 
                ? static_cast<mlir::Type>(getStringType()) 
                : static_cast<mlir::Type>(getNumberType());
            CAST(leftExpressionValue, location, type, leftExpressionValue, genContext);
        }

        if (MLIRTypeCore::canHaveToPrimitiveMethod(rightExpressionValue.getType())
            && evaluateProperty(location, rightExpressionValue, SYMBOL_TO_PRIMITIVE, genContext)
            && !isa<mlir_ts::UndefinedType>(leftExpressionValue.getType())
            && !isa<mlir_ts::NullType>(leftExpressionValue.getType()))
        {
            auto type = isa<mlir_ts::StringType>(leftExpressionValue.getType()) 
                ? static_cast<mlir::Type>(getStringType()) 
                : static_cast<mlir::Type>(getNumberType());
            CAST(rightExpressionValue, location, type, rightExpressionValue, genContext);
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

            // TODO: should it be int type especially PercentToken?
            if (leftExpressionValue.getType() != getNumberType())
            {
                CAST(leftExpressionValue, location, getNumberType(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != getNumberType())
            {
                CAST(rightExpressionValue, location, getNumberType(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::PlusToken:
        {
            // this is exactly the untyped default: case below (left/right type sync,
            // string-preferring) -- PlusToken used to fall through to it unconditionally.
            // Preserved as-is so string concat (`"fo" + 1`) and ordinary numeric-literal
            // widening (`numberParam + 1`) keep working exactly like before.
            auto leftType = leftExpressionValue.getType();
            if (isa<mlir_ts::StringType>(rightExpressionValue.getType()))
            {
                leftType = rightExpressionValue.getType();
                if (leftType != leftExpressionValue.getType())
                {
                    CAST(leftExpressionValue, location, leftType, leftExpressionValue, genContext);
                }
            }

            auto rightType = rightExpressionValue.getType();
            if (leftType != rightType)
            {
                CAST(rightExpressionValue, location, leftType, rightExpressionValue, genContext);
            }

            // additionally: when neither side is a string (so this isn't concat) and
            // both sides already had the SAME boolean type, the sync above was a no-op
            // (leftType == rightType already), so booleans reached
            // ArithmeticBinaryOpLowering as raw i1 and wrapped (`true + true` -> false
            // instead of 2). Widen them to number in that case.
            if (!isa<mlir_ts::StringType>(leftExpressionValue.getType()) && !isa<mlir_ts::StringType>(rightExpressionValue.getType()))
            {
                if (isa<mlir_ts::BooleanType>(leftExpressionValue.getType()))
                {
                    CAST(leftExpressionValue, location, getNumberType(), leftExpressionValue, genContext);
                }

                if (isa<mlir_ts::BooleanType>(rightExpressionValue.getType()))
                {
                    CAST(rightExpressionValue, location, getNumberType(), rightExpressionValue, genContext);
                }
            }

            break;
        }
        case SyntaxKind::MinusToken:
            // unlike PlusToken, MinusToken never does string concat, so it's safe to
            // widen booleans here and then fall through to the same cross-type sync
            // used by the other arithmetic/comparison operators below (e.g. `any - number`).
            if (isa<mlir_ts::BooleanType>(leftExpressionValue.getType()))
            {
                CAST(leftExpressionValue, location, getNumberType(), leftExpressionValue, genContext);
            }

            if (isa<mlir_ts::BooleanType>(rightExpressionValue.getType()))
            {
                CAST(rightExpressionValue, location, getNumberType(), rightExpressionValue, genContext);
            }

            [[fallthrough]];
        case SyntaxKind::AsteriskToken:
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
                // TODO: do we need to sync type for all Ops?
                static SmallVector<mlir::Type> types = {
                    builder.getF128Type(), 
                    getNumberType(), builder.getF64Type(), builder.getI64Type(), SInt(64), builder.getIndexType(),
                    builder.getF32Type(), SInt(32), builder.getI32Type(), 
                    builder.getF16Type(), SInt(16), builder.getI16Type(), 
                    SInt(8), builder.getI8Type()
                };

                auto r = syncUnionTypes(location, leftExpressionValue, rightExpressionValue, genContext);
                if (r.value)
                {                    
                    break;
                }

                if (mlir::failed(r.result))
                {
                    return mlir::failure();
                }
                
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
            auto leftType = leftExpressionValue.getType();

            // adjust left type
            if (isa<mlir_ts::StringType>(rightExpressionValue.getType()))
            {
                leftType = rightExpressionValue.getType();
                if (leftType != leftExpressionValue.getType())
                {
                    CAST(leftExpressionValue, location, leftType, leftExpressionValue, genContext);
                }
            }
            
            // sync right type to left type
            auto rightType = rightExpressionValue.getType(); 
            if (leftType != rightType)
            {
                CAST(rightExpressionValue, location, leftType, rightExpressionValue, genContext);
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

    std::string opName(SyntaxKind opCode) 
    {
        switch (opCode)
        {
            case SyntaxKind::PlusToken: return "plus";
            case SyntaxKind::MinusToken: return "minus";
            case SyntaxKind::AsteriskToken: return "multiply";
            case SyntaxKind::LessThanLessThanToken: return "leftShift";
            case SyntaxKind::GreaterThanGreaterThanToken: return "rightShift";
            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken: return "rightShiftUnsigned";
            case SyntaxKind::AmpersandToken: return "and";
            case SyntaxKind::BarToken: return "or";
            case SyntaxKind::CaretToken: return "xor";
            case SyntaxKind::EqualsToken: return "equals";
            case SyntaxKind::EqualsEqualsToken: return "equals";
            case SyntaxKind::EqualsEqualsEqualsToken: return "equals";
            case SyntaxKind::ExclamationEqualsToken: return "notEquals";
            case SyntaxKind::ExclamationEqualsEqualsToken: return "notEquals";
            case SyntaxKind::GreaterThanToken: return "greaterThan";
            case SyntaxKind::GreaterThanEqualsToken: return "greaterThanOrEquals";
            case SyntaxKind::LessThanToken: return "lessThan";
            case SyntaxKind::LessThanEqualsToken: return "lessThanOrEquals";
        default:
            return std::to_string((int)opCode);
            break;
        }
    }   

    ValueOrLogicalResult binaryOpLogicForUnions(mlir::Location location, SyntaxKind opCode, mlir::Value leftExpressionValue,
        mlir::Value rightExpressionValue, const GenContext &genContext)
    {
        if (leftExpressionValue && rightExpressionValue)
            if (auto leftUnionType = dyn_cast<mlir_ts::UnionType>(leftExpressionValue.getType()))
            {
                if (auto rightUnionType = dyn_cast<mlir_ts::UnionType>(rightExpressionValue.getType()))
                {
                    mlir::Type baseTypeLeft;
                    if (mth.isUnionTypeNeedsTag(location, leftUnionType, baseTypeLeft))
                    {
                        mlir::Type baseTypeRight;
                        if (mth.isUnionTypeNeedsTag(location, rightUnionType, baseTypeRight))
                        {
                            // info, we add "_" extra as scanner append "_" in front of "__";
                            auto funcName = "___bin_op_" + opName(opCode);

                            // we need to remove current implementation as we have different implementation per union type
                            removeGenericFunctionMap(funcName);

                            SmallVector<mlir::Type> classInstancesLeft;
                            for (auto subType : leftUnionType.getTypes())
                            {
                                mlir::TypeSwitch<mlir::Type>(subType)
                                    .Case<mlir_ts::ClassType>([&](auto classType_) { classInstancesLeft.push_back(classType_); })
                                    .Default([&](auto type) { 
                                    });                                   
                            }                            

                            SmallVector<mlir::Type> classInstancesRight;
                            for (auto subType : rightUnionType.getTypes())
                            {
                                mlir::TypeSwitch<mlir::Type>(subType)
                                    .Case<mlir_ts::ClassType>([&](auto classType_) { classInstancesRight.push_back(classType_); })
                                    .Default([&](auto type) { 
                                    });                                   
                            }                            

                            TypeOfOpHelper toh(builder);
                            
                            // TODO: must be improved
                            stringstream ss;

                            ss << S("function __bin_op_") << stows(opName(opCode)) << S("<L, R>(l: L, r: R) {\n");

                            auto printRightPart = [&] () {
                                for (auto rightSubType : rightUnionType.getTypes())
                                {
                                    auto typeOfNameRight = toh.typeOfAsString(rightSubType);
                                    ss << S("if (typeof(r) == \"") << stows(typeOfNameRight) << S("\") ");

                                    if (typeOfNameRight == "class")
                                    {
                                        ss << S("{\n");
                                        for (auto [index, _] : enumerate(classInstancesRight))
                                        {
                                            ss << S("if (r instanceof TYPE_INST_RIGHT_ALIAS");
                                            ss << index;
                                            ss << S(") return ") << S("l ") << Scanner::tokenStrings[opCode] << S(" r;\n");
                                        }                                        
                                        ss << S("}\n");
                                    }
                                    else
                                        ss << S("return ") << S("l ") << Scanner::tokenStrings[opCode] << S(" r;\n");
                                }
                            };

                            for (auto leftSubType : leftUnionType.getTypes())
                            {
                                auto typeOfNameLeft = toh.typeOfAsString(leftSubType);
                                ss << S("if (typeof(l) == \"") << stows(typeOfNameLeft) << S("\") {\n");
                                if (typeOfNameLeft == "class")
                                {
                                    for (auto [index, _] : enumerate(classInstancesLeft))
                                    {
                                        ss << S("if (l instanceof TYPE_INST_LEFT_ALIAS");
                                        ss << index;
                                        ss << S(") {\n");
                                        printRightPart();
                                        ss << S("}\n");
                                    }                                        
                                }
                                else
                                {
                                    printRightPart();
                                }

                                ss << S("}\n");
                            }

                            ss << "\nthrow \"Can't perform Binary Op for union types\";\n";                    
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
                            funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_L", leftUnionType});
                            funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_R", rightUnionType});

                            for (auto [index, instanceOfType] : enumerate(classInstancesLeft))
                            {
                                funcCallGenContext.typeAliasMap.insert({"TYPE_INST_LEFT_ALIAS" + std::to_string(index), instanceOfType});
                            }

                            for (auto [index, instanceOfType] : enumerate(classInstancesRight))
                            {
                                funcCallGenContext.typeAliasMap.insert({"TYPE_INST_RIGHT_ALIAS" + std::to_string(index), instanceOfType});
                            }

                            SmallVector<mlir::Value, 4> operands;
                            operands.push_back(leftExpressionValue);
                            operands.push_back(rightExpressionValue);

                            NodeFactory nf(NodeFactoryFlags::None);
                            return mlirGenCallExpression(
                                location, 
                                funcResult, 
                                { 
                                    nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_L")).as<Node>()), 
                                    nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_R")).as<Node>()) 
                                }, 
                                operands, 
                                funcCallGenContext);

                        }
                    }
                }
            }

        return mlir::success();
    }

    mlir::LogicalResult instantiateGenericsForBinaryOp(mlir::Location location, mlir::Value &leftExpressionValue,
        mlir::Value &rightExpressionValue, const GenContext &genContext)
    {
        if (isGenericFunctionReference(rightExpressionValue))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate function from generic: "
                                    << rightExpressionValue.getType() << " to match " << leftExpressionValue.getType() << "\n";);
            auto result = instantiateSpecializedFunction(
                location, rightExpressionValue, leftExpressionValue.getType(), genContext);
            if (mlir::failed(result))
            {
                return result;
            }

            auto resultValue = V(result);
            if (resultValue)
            {
                rightExpressionValue = resultValue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiated function: "
                                    << rightExpressionValue << "\n";);

        }      
        
        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(SpreadElement spreadElement, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(QualifiedName qualifiedName, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, bool isConditional,
                                                         const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         mlir::Value argument/*for index access*/,
                                                         const GenContext &genContext);

    ValueOrLogicalResult mlirGenPropertyAccessExpressionLogic(mlir::Location location, mlir::Value objectValue,
                                                              bool isConditional, MLIRPropertyAccessCodeLogic &cl,
                                                              const GenContext &genContext);

    mlir_ts::AccessLevel detectAccessLevel(mlir_ts::ClassStorageType classStorageType, const GenContext &genContext)
    {
        if (auto classInfo = getClassInfoByFullName(classStorageType.getName().getValue()))
        {
            return detectAccessLevel(classInfo->classType, genContext);
        }                    

        return mlir_ts::AccessLevel::Public;
    }    

    mlir_ts::AccessLevel detectAccessLevel(mlir_ts::ClassType classType, const GenContext &genContext)
    {
        auto accessingFromLevel = mlir_ts::AccessLevel::Public;
        if (genContext.thisClassType) {
            LLVM_DEBUG(llvm::dbgs() << "\n\t scope type \t'" << genContext.thisClassType << "' \n\t accessing type: \t" << classType << "\n";);

            if (genContext.thisClassType == classType) {
                accessingFromLevel = mlir_ts::AccessLevel::Private;
            } else {
                // check if protected level
                if (auto classInfo = getClassInfoByFullName(genContext.thisClassType.getName().getValue()))
                {
                    if (classInfo->hasBase(classType)) {
                        accessingFromLevel = mlir_ts::AccessLevel::Protected;
                    }
                }                    
            }
        }

        return accessingFromLevel;
    }

    ValueOrLogicalResult mlirGenPropertyAccessExpressionBaseLogic(mlir::Location location, mlir::Value objectValue,
                                                                  MLIRPropertyAccessCodeLogic &cl,
                                                                  const GenContext &genContext);

    mlir::Value extensionFunctionLogic(mlir::Location location, mlir::Value funcRef, mlir::Value thisValue, StringRef name,
                                  const GenContext &genContext);

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

            auto selectedNamespace = currentNamespace;

            while (selectedNamespace)
            {
                // search in outer namespaces
                while (selectedNamespace->isFunctionNamespace)
                {
                    selectedNamespace = selectedNamespace->parentNamespace;
                }

                for (auto &selectedNamespace : selectedNamespace->namespacesMap)
                {
                    if (selectedNamespace.getValue()->isFunctionNamespace)
                    {
                        continue;
                    }

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

                selectedNamespace = selectedNamespace->parentNamespace;
            }
        }        

        return mlir::Value();
    }

    mlir::Value ClassMembersAccess(mlir::Location location, mlir::Value thisValue, mlir::StringRef classFullName,
                             mlir::StringRef name, bool baseClass, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

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
                MLIRCodeLogic mcl(builder, compileOptions);
                thisValue = mcl.GetReferenceFromValue(location, thisValue);
                assert(thisValue);
            }

            CAST(effectiveThisValue, location, classType, thisValue, genContext);
        }        

        return effectiveThisValue;
    }

    mlir::Value ClassStaticFieldAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int staticFieldIndex, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext) {

        auto fieldInfo = classInfo->staticFields[staticFieldIndex];
        if (accessingFromLevel < fieldInfo.accessLevel) {
            emitError(location, "Class member ") << fieldInfo.id << " is not accessable";
            return mlir::Value();
        }

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
            mlir::Location location, mlir::Value thisValue, int methodIndex, bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext) {

        LLVM_DEBUG(llvm::dbgs() << "\n!! method index access: " << methodIndex << "\n";);

        auto methodInfo = classInfo->methods[methodIndex];
        if (accessingFromLevel < methodInfo.accessLevel) {
            emitError(location, "Class member '") << methodInfo.name << "' is not accessable";
            return mlir::Value();
        }

        StringRef funcName = methodInfo.funcName;
        auto effectiveFuncType = methodInfo.funcType;

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
                    auto globalFuncVar = resolveFullNameIdentifier(location, funcName, false, genContext);
                    if (!globalFuncVar)
                    {
                        emitError(location, "Class member '") << funcName << "' can't be resolved (dynamic import)";
                        return mlir::Value();
                    }

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
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));                            
                        
                        return thisSymbOp;
                    }

                    auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                        location, effectiveFuncType,
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
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
                mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
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
                LLVM_DEBUG(dbgs() << "\n!! Virtual call: func '" << funcName << "'\n";);

                LLVM_DEBUG(dbgs() << "\n!! Virtual call - this val: [ " << effectiveThisValue << " ] func type: [ "
                                    << effectiveFuncType << " ] isStorage access: " << isStorageType << "\n";);

                // auto inTheSameFunc = funcName == const_cast<GenContext &>(genContext).funcName;

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
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
                return thisVirtualSymbOp;
            }

            if (classInfo->isDynamicImport)
            {
                // Direct (non-virtual) access to a dynamic-import class member - e.g. a
                // cross-module `super.method(...)` call, or a non-virtual inherited method.
                // Resolution order:
                //
                // 1. A FuncOp WITH a body: the method is actually defined in this module
                //    (compiler-synthesized methods like .instanceOf get real FuncOps even
                //    for isDynamicImport classes - see mlirGenClassInstanceOfMethod). Only
                //    a defined body qualifies: a bodyless declaration FuncOp would lower
                //    to a plain external symbol reference, which the dynamic import mode
                //    (-shared without an import .lib) cannot link.
                if (auto funcOp = theModule.lookupSymbol<mlir_ts::FuncOp>(funcName))
                {
                    if (!funcOp.getBody().empty())
                    {
                        auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                            location, getBoundFunctionType(effectiveFuncType), effectiveThisValue,
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
                        return thisSymbOp;
                    }
                }

                // 2. The dlsym-style global variable mlirGenClassMethodMemberDynamicImport /
                //    mlirGenFunctionLikeDeclarationDynamicImport registered for @dllimport
                //    members (statics/constructors/.new today).
                auto globalFuncVar = resolveFullNameIdentifier(location, funcName, false, genContext);
                if (!globalFuncVar)
                {
                    // 3. Inline dlsym. Compiler-synthesized methods (.instanceOf) and plain
                    //    instance methods of an imported class have neither of the above: no
                    //    per-member @dllimport decorator ever routes them through the
                    //    registration path, and their FuncOp (when one exists at all) is a
                    //    bodyless declaration. Registering a global lazily from HERE is not
                    //    an option either - this can run inside a transient discovery scope
                    //    ("simulate scope"), where a fullNameGlobalsMap registration is torn
                    //    down with the scope, or worse trips its LIFO assert; see
                    //    docs/cross-module-dynamic-import-instanceof-design.md §5-§7 for the
                    //    two reverted attempts. So resolve the symbol in place, exactly like
                    //    the registered variant's initializer does
                    //    (mlirGenFunctionLikeDeclarationDynamicImport): the DLL is already
                    //    loaded by the import's LoadLibraryPermanentlyOp global ctor by the
                    //    time any method body runs. Self-contained: no global state, valid
                    //    in both discovery (ops land in the throwaway module) and real
                    //    passes, at the cost of one symbol lookup per call site.
                    auto symbolNameValue = V(mlirGenStringValue(location, funcName.str(), true));
                    auto referenceToFuncOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                        location, getOpaqueType(), symbolNameValue);
                    auto castResult = cast(location, effectiveFuncType, referenceToFuncOpaque, genContext);
                    if (castResult.failed_or_no_value())
                    {
                        emitError(location, "Class member '") << funcName << "' can't be resolved (dynamic import)";
                        return mlir::Value();
                    }

                    globalFuncVar = V(castResult);
                }

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
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
                return thisSymbOp;
            }
        }
    }

    mlir::Value ClassGenericMethodAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int genericMethodIndex, 
            bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

    mlir::Value ClassAccessorAccess(ClassInfo::TypePtr classInfo,
            mlir::Location location, mlir::Value thisValue, int accessorIndex,
            bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

    mlir::Value ClassIndexAccess(ClassInfo::TypePtr classInfo,
            mlir::Location location, mlir::Value thisValue, mlir::Value argument, bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

    mlir::Value ClassBaseClassAccess(ClassInfo::TypePtr classInfo, ClassInfo::TypePtr baseClass, int index,
            mlir::Location location, mlir::Value thisValue, StringRef name, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

    mlir::Value ClassMembersAccess(mlir::Location location, mlir::Value thisValue, ClassInfo::TypePtr classInfo,
                             mlir::StringRef name, bool isSuperClass, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext);

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
                                 mlir::Attribute id, mlir::Value argument, const GenContext &genContext);

    // vtableOffset is the declaring interface's slot-block position within the root
    // interface's combined vtable (0 unless fieldInfo was found through an `extends` chain
    // - see InterfaceInfo::findField's doc comment, MLIRGenStore.h).
    mlir::Value InterfaceFieldAccess(mlir::Location location, mlir::Value interfaceValue, InterfaceFieldInfo *fieldInfo, int vtableOffset = 0)
    {
        auto fieldRefType = mlir_ts::RefType::get(fieldInfo->type);

        // fieldInfo->virtualIndex is assigned once, canonically, by
        // InterfaceInfo::assignCanonicalVirtualIndexes() when the interface declaration
        // resolves - it is never -1 here. Whether THIS PARTICULAR interface value's
        // underlying object actually provides an optional member is a runtime property (it
        // can differ between implementers of the same interface), not something this call
        // site can know at compile time - InterfaceSymbolRefOpLowering's isOptional branch
        // (LowerToLLVM.cpp) already checks the loaded slot against the -1 sentinel and
        // produces OptionalUndef at runtime when appropriate. An earlier version of this
        // function special-cased virtualIndex == -1 here to bypass the runtime read - that
        // relied on InterfaceInfo::getVirtualTable() mutating the SHARED virtualIndex to -1
        // as a side effect of whichever cast last happened to be missing this member, which
        // corrupted access sites for OTHER, unrelated implementers that do provide it. See
        // docs/interface-vtable-simplification-design.md §5.
        assert(fieldInfo->virtualIndex >= 0);
        auto vtableIndex = vtableOffset + fieldInfo->virtualIndex;

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

    // see InterfaceFieldAccess's vtableOffset doc comment above.
    mlir::Value InterfaceMethodAccess(mlir::Location location, mlir::Value interfaceValue, InterfaceMethodInfo *methodInfo, int vtableOffset = 0)
    {
        assert(methodInfo->virtualIndex >= 0);
        auto vtableIndex = vtableOffset + methodInfo->virtualIndex;

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
            int vtableOffset;
            if (auto getMethodInfo = interfaceInfo->findMethod(accessorInfo->getMethod, vtableOffset))
            {
                getMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, getMethodInfo, vtableOffset);
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
            int vtableOffset;
            if (auto setMethodInfo = interfaceInfo->findMethod(accessorInfo->setMethod, vtableOffset))
            {
                setMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, setMethodInfo, vtableOffset);
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
            int vtableOffset;
            if (auto getMethodInfo = interfaceInfo->findMethod(indexInfo->getMethod, vtableOffset))
            {
                getMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, getMethodInfo, vtableOffset);
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
            int vtableOffset;
            if (auto setMethodInfo = interfaceInfo->findMethod(indexInfo->setMethod, vtableOffset))
            {
                setMethodInfoValue = InterfaceMethodAccess(location, interfaceValue, setMethodInfo, vtableOffset);
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
        mlir::Attribute id, mlir::Value argument, const GenContext &genContext);

    template <typename T>
    ValueOrLogicalResult mlirGenElementAccessTuple(mlir::Location location, mlir::Value expression,
                                              mlir::Value argumentExpression, T tupleType)
    {
        // get index
        if (auto indexConstOp = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
        {
            // this is property access
            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, expression, indexConstOp.getValue());
            return cl.Tuple(tupleType, true);
        }
        else
        {
            // tuples/object-literals here compile to a fixed-layout struct (their fields
            // are resolved to specific offsets at compile time), not a dynamic hash map -
            // a runtime-computed (non-constant) index genuinely can't resolve to a
            // specific field, so this is a real language limitation, not a missing code
            // path. Fail with a clear diagnostic instead of crashing.
            emitError(location) << "Element access with a non-constant index is not supported on this type; "
                                    "only array types and constant keys (obj[\"literal\"]) can be indexed";
            return ValueOrLogicalResult(mlir::failure());
        }
    }

    ValueOrLogicalResult mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGenElementAccess(mlir::Location location, mlir::Value expression, mlir::Value argumentExpression, bool isConditionalAccess, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(CallExpression callExpression, const GenContext &genContext);

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
                                            const GenContext &genContext);

    ValueOrLogicalResult mlirGenCallBuiltInFunction(
        mlir::Location location, mlir::Value actualFuncRefValue, NodeArray<TypeNode> typeArguments, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext);

    ValueOrLogicalResult mlirGenCallExpression(mlir::Location location, mlir::Value funcResult,
                                               NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                               const GenContext &genContext);

    ValueOrLogicalResult NewClassInstanceOnStack(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext);

    ValueOrLogicalResult NewClassInstance(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext, bool onStack = false);

    ValueOrLogicalResult mlirGenCall(mlir::Location location, mlir::Value funcRefValue,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext);

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
                //LLVM_DEBUG(llvm::dbgs() << "\t last value = " << operands.back() << "\n";);

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
            : operands{operands}, lastArgIndex{-1}, hasType{false}, hasVarArgs{false}, currentParameter{offsetArgs}, 
              noReceiverTypesForGenericCall{noReceiverTypesForGenericCall}, noCastNeeded{false}, mth{mth}
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
                        noCastNeeded = true;
                    }
                }
                else
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! VarArg type is: " << varArgType << "\n";);
                    // in case of generics which are not defined yet, array will be identified later in generic method call
                    varArgType = mlir::Type();
                    hasVarArgs = false;
                    noCastNeeded = true;
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
            if (noCastNeeded)
            {
                return mlir::Type();
            }

            auto receiverType = getReceiverType();
            if (isOptionalUnwrap && receiverType) 
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
        bool noCastNeeded;
        MLIRTypeHelper &mth;
    };

    ValueOrLogicalResult callIteratorNext(mlir::Location location, mlir::Value nextProperty, 
        OperandsProcessingInfo* operandsProcessingInfo, const GenContext &genContext);

    bool hasIterator(mlir::Location location, mlir::Value source, const GenContext &genContext) 
    {
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
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { 
                        fields = tupleType.getFields(); 
                    })
                    .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { 
                        fields = constTupleType.getFields(); 
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                auto propValue = mlir::StringAttr::get(builder.getContext(), "value");
                if (std::any_of(fields.begin(), fields.end(), [&] (auto field) { return field.id == propValue; }))
                {
                    return true;
                }
            }
        }     
        
        return false;
    }

    bool isArrayLike(mlir::Location location, mlir::Value source, const GenContext &genContext) 
    {
        if (auto lengthPropertyType = evaluateProperty(location, source, LENGTH_FIELD_NAME, genContext))
        {
            return true;
        }

        return false;
    }

    mlir::LogicalResult processOperandSpreadElement(mlir::Location location, mlir::Value source, OperandsProcessingInfo &operandsProcessingInfo, const GenContext &genContext);

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
        MLIRCodeLogic mcl(builder, compileOptions);
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
                                               bool castThisValueToClass, const GenContext &genContext);

    // TODO: refactor it, somehow when NewClassInstanceAsMethodCallOp calling Ctor and NewClassInstanceLogicAsOp is not
    ValueOrLogicalResult NewClassInstance(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
                                          NodeArray<TypeNode> typeArguments, bool suppressConstructorCall, 
                                          const GenContext &genContext);

    ValueOrLogicalResult NewClassInstanceLogicAsOp(mlir::Location location, mlir::Type typeOfInstance, bool stackAlloc,
                                                   const GenContext &genContext);

    mlir::Value NewClassInstanceLogicAsOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool stackAlloc,
                                          const GenContext &genContext);

    mlir::Value NewClassInstanceAsMethodCallOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool asMethodCall,
                                             const GenContext &genContext);

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
            count = builder.create<mlir_ts::ConstantOp>(location, builder.getIndexType(), builder.getIndexAttr(0));
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

        if (count.getType() != builder.getIndexType())
        {
            // TODO: test cast result
            count = cast(location, builder.getIndexType(), count, genContext);
        }

        auto newArrOp = builder.create<mlir_ts::NewArrayOp>(location, getArrayType(elementType), count);
        return V(newArrOp);                     
    }    

    ValueOrLogicalResult NewClassInstanceByCallingNewCtor(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
            NodeArray<TypeNode> typeArguments, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(NewExpression newExpression, const GenContext &genContext);

    mlir::LogicalResult mlirGen(DeleteExpression deleteExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(VoidExpression voidExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(NonNullExpression nonNullExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(OmittedExpression ommitedExpression, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(TaggedTemplateExpression taggedTemplateExpressionAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(NullLiteral nullLiteral, const GenContext &genContext);

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

    ValueOrLogicalResult mlirGen(TrueLiteral trueLiteral, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(FalseLiteral falseLiteral, const GenContext &genContext);

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

    ValueOrLogicalResult mlirGen(NumericLiteral numericLiteral, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext);

    ValueOrLogicalResult mlirGenStringValue(mlir::Location location, StringRef text, bool asString = false)
    {
        auto attrVal = getStringAttr(text);
        auto literalType = asString ? (mlir::Type)getStringType() : (mlir::Type)mlir_ts::LiteralType::get(attrVal, getStringType());
        return V(builder.create<mlir_ts::ConstantOp>(location, literalType, attrVal));
    }

    ValueOrLogicalResult mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ts::RegularExpressionLiteral regularExpressionLiteral, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral,
                                 const GenContext &genContext);

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
            MLIRTypeHelper mth(nullptr, {});
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

        if (auto nextPropertyType = evaluateProperty(location, itemValue, ITERATOR_NEXT, genContext))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadElement, next type is: " << nextPropertyType << "\n";);

            if (auto returnType = mth.getReturnTypeFromFuncRef(nextPropertyType))
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
                    MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, itemValue, builder.getIndexAttr(++index));
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

        // ArrayLike, do not put it before Tuple & Const Tuple, otherwise [xxx] will return wrong type
        if (auto indexAccessType = evaluateElementAccess(location, itemValue, false, genContext))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadElement, [number] type is: " << indexAccessType << "\n";);

            values.push_back({itemValue, true, true});

            accumulateArrayItemType(location, indexAccessType, arrayInfo);

            return mlir::success();            
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! spread element type: " << type << "\n";);
        emitError(location, "can't estimate element of array");

        return mlir::failure();
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
                fieldInfos.push_back({mlir::Attribute(), type, false, mlir_ts::AccessLevel::Public});
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
            fieldInfos.push_back({mlir::Attribute(), val.value.getType(), false, mlir_ts::AccessLevel::Public});
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

                cm.mlirGenArrayPush(
                    location, 
                    loadedVarArray, 
                    vals,
                    [this](mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext, bool disableStrictNullCheck) { return cast(location, type, value, genContext, disableStrictNullCheck); }, 
                    genContext);
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

    ValueOrLogicalResult mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext);

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

    // accumulated state for building an object literal (fields, methods, captures)
    struct ObjectLiteralInfo
    {
        ts::ObjectLiteralExpression objectLiteral;
        mlir_ts::ObjectType objThis;
        mlir::Type receiverType;
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        SmallVector<size_t> methodInfos;
        SmallVector<std::pair<std::string, size_t>> methodInfosWithCaptures;
        SmallVector<std::pair<mlir::Attribute, mlir::Value>> fieldsToSet;
    };

    mlir::LogicalResult addObjectFuncFieldInfo(ObjectLiteralInfo &oli, mlir::Attribute fieldId, const std::string &funcName,
                                               mlir_ts::FunctionType funcType)
    {
        auto type = funcType;

        oli.values.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
        oli.fieldInfos.push_back({fieldId, type, false, mlir_ts::AccessLevel::Public});

        // record (this literal's object-storage type, field name) -> the lifted function's
        // symbol so a later interface vtable build (mlirGenObjectVirtualTableDefinitionForInterface)
        // can emit a constant SymbolRefOp for this method slot instead of the runtime
        // load-from-object patch - see docs/interface-vtable-simplification-design.md §3.
        // oli.objThis.getStorageType() is the SAME (hash-consed) ObjectStorageType that ends
        // up embedded as the "this" parameter of this field's own FunctionType, which is how
        // the vtable builder recovers this key later even though by then it only has the
        // BOXED object type (whose top-level storage is a plain TupleType, not this
        // ObjectStorageType directly - see the design doc's §3 implementation notes for why
        // that indirection is necessary, not optional). Captures are fine here too: a
        // captures-bearing method's field value is still this same compile-time-constant
        // funcName (the per-instance data lives in a separate `.captured` field the function
        // reads via `this`, not in a different function per instance).
        objectLiteralMethodSymbolsMap[oli.objThis.getStorageType()][fieldId] = funcName;

        if (getCaptureVarsMap().find(funcName) != getCaptureVarsMap().end())
        {
            oli.methodInfosWithCaptures.push_back({funcName, oli.fieldInfos.size() - 1});
        }
        else
        {
            oli.methodInfos.push_back(oli.fieldInfos.size() - 1);
        }

        return mlir::success();
    }

    mlir::LogicalResult addObjectFieldInfoToArrays(ObjectLiteralInfo &oli, mlir::Attribute fieldId, mlir::Type type)
    {
        oli.values.push_back(builder.getUnitAttr());
        oli.fieldInfos.push_back({fieldId, type, false, mlir_ts::AccessLevel::Public});

        return mlir::success();
    }

    mlir::LogicalResult addObjectFieldInfo(mlir::Location location, ObjectLiteralInfo &oli, mlir::Attribute fieldId,
                                           mlir::Value itemValue, mlir::Type receiverElementType, const GenContext &genContext)
    {
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
            if (type != receiverElementType)
            {
                value = builder.getUnitAttr();
                itemValue = cast(location, receiverElementType, itemValue, genContext);
                isConstValue = false;
            }

            type = receiverElementType;
        }

        oli.values.push_back(value);
        oli.fieldInfos.push_back({fieldId, type, false, mlir_ts::AccessLevel::Public});
        if (!isConstValue)
        {
            oli.fieldsToSet.push_back({fieldId, itemValue});
        }

        return mlir::success();
    }

    mlir::LogicalResult processObjectFunctionLikeProto(ObjectLiteralInfo &oli, mlir::Attribute fieldId,
                                                       FunctionLikeDeclarationBase funcLikeDecl, const GenContext &genContext)
    {
        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        funcGenContext.clearReceiverTypes();
        funcGenContext.thisType = oli.objThis;

        funcLikeDecl->parent = oli.objectLiteral;

        // generator methods/properties must resolve to the generator wrapper type (the object with
        // `.next()`, not the bare yielded-value tuple), same as mlirGenFunctionLikeDeclaration does for
        // top-level function* and class generator methods; otherwise the field type registered here
        // (used for the object literal's own type) is wrong and later access like `obj.gen().next()`
        // fails to resolve.
        auto protoDecl = funcLikeDecl->asteriskToken
            ? buildGeneratorWrapperDeclaration(funcLikeDecl, loc(funcLikeDecl))
            : funcLikeDecl;

        auto [funcOp, funcProto, result, isGeneric] = mlirGenFunctionPrototype(protoDecl, funcGenContext);
        if (mlir::failed(result) || !funcOp)
        {
            return mlir::failure();
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
                    addObjectFieldInfoToArrays(oli, fieldInfo.id, fieldInfo.type);
                }
            }
        }

        return addObjectFuncFieldInfo(oli, fieldId, funcName, funcType);
    }

    mlir::LogicalResult processObjectFunctionLike(ObjectLiteralInfo &oli, FunctionLikeDeclarationBase funcLikeDecl,
                                                  const GenContext &genContext)
    {
        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        funcGenContext.clearReceiverTypes();
        funcGenContext.thisType = oli.objThis;

        LLVM_DEBUG(llvm::dbgs() << "\n!! Object Process function with this type: " << oli.objThis << "\n";);

        funcLikeDecl->parent = oli.objectLiteral;

        // this method's function pointer is reachable only indirectly, baked
        // into the exported object literal's data - force public/external
        // linkage (same mechanism exported class methods use, see
        // MLIRGenClasses.cpp) so the linker doesn't strip it as unreferenced.
        if (genContext.isExportVarReceiver)
        {
            funcLikeDecl->internalFlags |= InternalFlags::IsPublic;
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        auto [result, funcOp, funcName, isGeneric] = mlirGenFunctionLikeDeclaration(funcLikeDecl, funcGenContext);
        return result;
    }

    // pass 1: add all data fields; engaged result = early exit (failure or partial-resolve no-value)
    std::optional<ValueOrLogicalResult> mlirGenObjectLiteralFields(mlir::Location location, ObjectLiteralInfo &oli,
                                                                   const GenContext &genContext)
    {
        for (auto &item : oli.objectLiteral->properties)
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

                if (oli.receiverType)
                {
                    receiverElementType = getTypeByFieldNameFromReceiverType(fieldId, oli.receiverType);
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
                if (oli.receiverType && !receiverElementType)
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! Detecting dest. union type with first field: " << fieldId << "\n";);

                    if (auto unionType = dyn_cast<mlir_ts::UnionType>(oli.receiverType))
                    {
                        for (auto subType : unionType.getTypes())
                        {
                            auto possibleType = getTypeByFieldNameFromReceiverType(fieldId, subType);
                            if (possibleType == itemValue.getType())
                            {
                                LLVM_DEBUG(llvm::dbgs() << "\n!! we picked type from union: " << subType << "\n";);

                                receiverElementType = possibleType;
                                oli.receiverType = subType;
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

                auto tupleFields = [&] (::llvm::ArrayRef<mlir_ts::FieldInfo> fields) -> mlir::LogicalResult {
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
                        if (mlir::failed(addObjectFieldInfo(
                            location,
                            oli,
                            std::get<0>(pair).id,
                            std::get<1>(pair),
                            oli.receiverType
                                ? getTypeByFieldNameFromReceiverType(std::get<0>(pair).id, oli.receiverType)
                                : mlir::Type(),
                            genContext))) {
                            return mlir::failure();
                        }
                    }

                    return mlir::success();
                };

                auto resultForTuple = mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(tupleValue.getType())
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { return tupleFields(tupleType.getFields()); })
                    .template Case<mlir_ts::ConstTupleType>(
                        [&](auto constTupleType) { return tupleFields(constTupleType.getFields()); })
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

                                        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, tupleValue, fieldInfo.id);
                                        // TODO: implemenet conditional
                                        mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, tupleValue, interfaceFieldInfo->isConditional, cl, genContext);
                                        if (mlir::failed(addObjectFieldInfo(location, oli, fieldInfo.id, propertyAccess, receiverElementType, genContext))) {
                                            return mlir::failure();
                                        }
                                    }
                                }
                            }

                            return mlir::success();
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

                                        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, tupleValue, fieldInfo.id);
                                        // TODO: implemenet conditional
                                        mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, tupleValue, false, cl, genContext);
                                        if (mlir::failed(addObjectFieldInfo(location, oli, fieldInfo.id, propertyAccess, receiverElementType, genContext))) {
                                            return mlir::failure();
                                        }
                                    }
                                }
                            }

                            return mlir::success();
                        })
                    .template Case<mlir_ts::ObjectType>(
                        [&](auto objectType) {
                            // boxed object literal (docs/object-literal-boxing-design.md): a
                            // pointer, not a value tuple, so read each field via property access
                            // (like ClassType) rather than DeconstructTupleOp (which needs a value).
                            mlir::SmallVector<mlir_ts::FieldInfo> destFields;
                            if (mlir::failed(mth.getFields(objectType, destFields)))
                            {
                                return mlir::failure();
                            }

                            for (auto fieldInfo : destFields)
                            {
                                MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, tupleValue, fieldInfo.id);
                                mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, tupleValue, false, cl, genContext);
                                if (mlir::failed(addObjectFieldInfo(location, oli, fieldInfo.id, propertyAccess, receiverElementType, genContext))) {
                                    return mlir::failure();
                                }
                            }

                            return mlir::success();
                        })
                    .Default([&](auto type) {
                        // spreading a tuple/const-tuple/interface/class/(boxed) object into an
                        // object literal is handled above; spreading anything else (an array,
                        // a union, a primitive...) is a real, unimplemented feature (e.g. array
                        // spread would need to synthesize numeric-string-keyed fields "0","1",...)
                        // rather than an unreachable state - fail with a clear diagnostic instead
                        // of crashing until it's implemented.
                        emitError(location) << "Spread in an object literal is not supported for type: " << to_print(type);
                        return mlir::failure();
                    });

                if (mlir::failed(resultForTuple)) {
                    return ValueOrLogicalResult(resultForTuple);
                }

                continue;
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1)");
            }

            assert(genContext.allowPartialResolve || itemValue);

            if (mlir::failed(addObjectFieldInfo(location, oli, fieldId, itemValue, receiverElementType, genContext))) {
                return ValueOrLogicalResult(mlir::failure());
            }
        }

        return std::nullopt;
    }

    // pass 2: register method prototypes as fields
    mlir::LogicalResult mlirGenObjectLiteralMethodPrototypes(ObjectLiteralInfo &oli, const GenContext &genContext)
    {
        for (auto &item : oli.objectLiteral->properties)
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
                if (mlir::failed(processObjectFunctionLikeProto(oli, fieldId, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
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
                if (mlir::failed(processObjectFunctionLikeProto(oli, fieldId, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
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

                if (mlir::failed(processObjectFunctionLikeProto(oli, fieldId, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    // accumulate captured vars of all methods into one '.captured' field
    mlir::LogicalResult mlirGenObjectLiteralCaptures(mlir::Location location, ObjectLiteralInfo &oli, const GenContext &genContext)
    {
        llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> accumulatedCaptureVars;

        for (auto &methodRefWithName : oli.methodInfosWithCaptures)
        {
            auto funcName = std::get<0>(methodRefWithName);
            auto methodRef = std::get<1>(methodRefWithName);
            auto &methodInfo = oli.fieldInfos[methodRef];

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
            if (mlir::failed(mlirGenResolveCapturedVars(location, accumulatedCaptureVars, accumulatedCapturedValues, genContext))) {
                return mlir::failure();
            }

            MLIRCodeLogic mcl(builder, compileOptions);
            auto capturedValue = mlirGenCreateCapture(location, mcl.CaptureType(accumulatedCaptureVars),
                                                      accumulatedCapturedValues, genContext);
            if (mlir::failed(addObjectFieldInfo(location, oli, MLIRHelper::TupleFieldName(CAPTURED_NAME, builder.getContext()), capturedValue, mlir::Type(), genContext))) {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    // pass 3: generate method bodies
    mlir::LogicalResult mlirGenObjectLiteralMethodBodies(ObjectLiteralInfo &oli, const GenContext &genContext)
    {
        for (auto &item : oli.objectLiteral->properties)
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
                if (mlir::failed(processObjectFunctionLike(oli, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
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
                if (mlir::failed(processObjectFunctionLike(oli, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
            }
            else if (item == SyntaxKind::MethodDeclaration || item == SyntaxKind::GetAccessor || item == SyntaxKind::SetAccessor)
            {
                auto funcLikeDecl = item.as<FunctionLikeDeclarationBase>();
                if (mlir::failed(processObjectFunctionLike(oli, funcLikeDecl, genContext))) {
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext);
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

    ValueOrLogicalResult mlirGen(Identifier identifier, const GenContext &genContext);

    mlir::Value resolveIdentifierAsVariable(mlir::Location location, StringRef name, const GenContext &genContext);

    mlir::LogicalResult mlirGenResolveCapturedVars(mlir::Location location,
                                                   llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> captureVars,
                                                   SmallVector<mlir::Value> &capturedValues,
                                                   const GenContext &genContext);

    ValueOrLogicalResult mlirGenCreateCapture(mlir::Location location, mlir::Type capturedType,
                                              SmallVector<mlir::Value> capturedValues, const GenContext &genContext);

    mlir::Value resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir_ts::FunctionType funcType,
                                           mlir::Value thisValue, bool addGenericAttrFlag,
                                           const GenContext &genContext);

    mlir::Value resolveFunctionNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext);

    mlir::Type resolveTypeByNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext);

    mlir::Type resolveTypeByName(mlir::Location location, StringRef name, const GenContext &genContext);

    mlir::Value resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext);

    mlir::Value resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess,
                                          const GenContext &genContext);

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

        auto loadOp = builder.create<mlir_ts::LoadOp>(location, value->getType(), address);
        if (value->getAtomic())
        {
            loadOp->setAttr(ATOMIC_ATTR_NAME, builder.getBoolAttr(true));
            loadOp->setAttr(ORDERING_ATTR_NAME, builder.getI32IntegerAttr(value->getOrdering()));
            loadOp->setAttr(SYNCSCOPE_ATTR_NAME, builder.getStringAttr(value->getSyncScope()));
        }

        if (value->getVolatile())
        {
            loadOp->setAttr(VOLATILE_ATTR_NAME, builder.getBoolAttr(true));
        }

        if (value->getNonTemporal())
        {
            loadOp->setAttr(NONTEMPORAL_ATTR_NAME, builder.getBoolAttr(true));
        }            

        if (value->getInvariant())
        {
            loadOp->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
        }   

        return loadOp;
    }

    mlir::Value resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(mlir::Location location, StringRef name, const GenContext &genContext);

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

    // mutates genContext.typeParamsWithArgs with type params zipped from the receiver function type
    mlir::LogicalResult processTypeArgumentsFromFunctionParameters(SignatureDeclarationBase signatureDeclarationBase,
                                              GenContext &genContext)
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
                    location, genContext.typeParamsWithArgs, typeParam, type, false, genContext);
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

                // support dynamic loading: like a generic class/function/interface, a
                // generic type alias has no compiled body or runtime entity at all -
                // resolution is pure compile-time type substitution
                // (getTypeByTypeReference/resolveGenericTypeInNamespace call
                // getType(typeNode, ...) directly on the stored TypeNode, never
                // re-invoking mlirGen on the whole declaration), unlike
                // GenericClassInfo/GenericFunctionInfo/GenericInterfaceInfo there is no
                // dedicated Info struct tracking sourceFile/elementNamespace for this
                // declaration kind - print inline here instead of adding one, since this
                // is the only place needing it and there is no "already registered"
                // early-return branch to retry from (unlike registerGenericClass/
                // registerGenericFunctionLike/registerGenericInterface):
                // isAddedToExport's own stage gate is unnecessary too since a given
                // top-level TypeAliasDeclaration node is visited exactly once per stage,
                // so a direct stage check is sufficient dedup.
                if (hasExportModifier && stage == Stages::SourceGeneration)
                {
                    addGenericTypeAliasDeclarationToExport(typeAliasDeclarationAST, currentNamespace);
                }
            }
            else
            {
                if (hasExportModifier)
                {
                    GenContext typeAliasGenContext(genContext);
                    auto type = getType(typeAliasDeclarationAST->type, typeAliasGenContext);
                    if (type)
                    {
                        getTypeAliasMap().insert({ namePtr, { type, undefined } });
                        addTypeDeclarationToExport(namePtr, currentNamespace, type);
                    }
                }
                else
                {
                    getTypeAliasMap().insert({ namePtr, { mlir::Type(), typeAliasDeclarationAST->type } });
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

    mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext);

    mlir::LogicalResult registerGenericClass(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext);

    mlir::LogicalResult mlirGen(ClassDeclaration classDeclarationAST, const GenContext &genContext);

    ValueOrLogicalResult mlirGen(ClassExpression classExpressionAST, const GenContext &genContext);

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
                                                            const GenContext &genContext);

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

    ClassInfo::TypePtr mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext);

    ClassInfo::TypePtr mlirGenClassInfo(const std::string &name, ClassLikeDeclaration classDeclarationAST,
                                        const GenContext &genContext);

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
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassTypeSetFields(ClassInfo::TypePtr newClassPtr,
                                                  SmallVector<mlir_ts::FieldInfo> &fieldInfos);

    mlir::LogicalResult mlirGenClassStorageType(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassStaticFields(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassMembers(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                            ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

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
                                                   const GenContext &genContext);

    Node getFieldNameForAccessor(Node name) {
        auto nameStr = MLIRHelper::getName(name);
        nameStr.insert(0, "#__");

        NodeFactory nf(NodeFactoryFlags::None);
        auto newName = nf.createIdentifier(stows(nameStr.c_str()));
        return newName;
    }

    mlir::LogicalResult mlirGenClassDataFieldAccessor(mlir::Location location, ClassInfo::TypePtr newClassPtr, 
            PropertyDeclaration propertyDeclaration, MemberName name, mlir::Type typeIfNotProvided, const GenContext &genContext);

    mlir_ts::AccessLevel getAccessLevel(Node node) {
        return hasModifier(node, SyntaxKind::PrivateKeyword) 
                ? mlir_ts::AccessLevel::Private 
                : hasModifier(node, SyntaxKind::ProtectedKeyword) 
                    ? mlir_ts::AccessLevel::Protected 
                    : mlir_ts::AccessLevel::Public;
    }

    mlir::LogicalResult mlirGenClassDataFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                    PropertyDeclaration propertyDeclaration, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassStaticFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassStaticFieldMemberDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassConstructorPublicDataFieldMembers(mlir::Location location, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                                      ConstructorDeclaration constructorDeclaration, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassProcessClassPropertyByFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember);

    mlir::LogicalResult mlirGenClassFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                SmallVector<mlir_ts::FieldInfo> &fieldInfos, bool staticOnly,
                                                const GenContext &genContext);

    mlir::LogicalResult mlirGenForwardDeclaration(const std::string &funcName, mlir_ts::FunctionType funcType,
                                                  bool isStatic, bool isVirtual, bool isAbstract,
                                                  ClassInfo::TypePtr newClassPtr, int orderWeight, 
                                                  mlir_ts::AccessLevel accessLevel, const GenContext &genContext)
    {
        if (newClassPtr->getMethodIndex(funcName) < 0)
        {
            return mlir::success();
        }

        newClassPtr->methods.push_back(
        {
            funcName,
            funcType,
            std::string(), // forward declaration: no function symbol yet
            isStatic,
            isVirtual || isAbstract, 
            isAbstract, 
            -1, 
            orderWeight, 
            accessLevel
        });        
        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassNew(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                        const GenContext &genContext);

    mlir::LogicalResult mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassDefaultStaticConstructor(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             const GenContext &genContext);

    // to support crearting classes in Stack
    mlir::LogicalResult mlirGenClassSizeStaticField(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    void pushStaticField(llvm::SmallVector<StaticFieldInfo> &staticFieldInfos, mlir::Attribute fieldId, mlir::Type staticFieldType, 
        StringRef fullClassStaticFieldName, int index, mlir_ts::AccessLevel accessLevel)
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
            staticFieldInfos.push_back({fieldId, staticFieldType, fullClassStaticFieldName, index, accessLevel});
        }        
    }

    // INFO: you can't use standart Static Field declarastion because of RTTI should be declared before used
    // example: C:/dev/TypeScriptCompiler/tslang/test/tester/tests/dependencies.ts
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

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, mlir_ts::AccessLevel::Public);

        return mlir::success();
    }

    // INFO: you can't use standart Static Field declarastion because of RTTI should be declared before used
    // example: C:/dev/TypeScriptCompiler/tslang/test/tester/tests/dependencies.ts
    mlir::LogicalResult mlirGenCustomRTTIDynamicImport(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        return mlirGenStaticFieldDeclarationDynamicImport(location, newClassPtr, RTTI_NAME, getStringType(), mlir_ts::AccessLevel::Public, genContext);
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
                                                        const GenContext &genContext);

    mlir::LogicalResult mlirGenClassTypeBitmap(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                               const GenContext &genContext);

#endif

    mlir::LogicalResult mlirGenClassInstanceOfMethod(ClassLikeDeclaration classDeclarationAST,
                                                     ClassInfo::TypePtr newClassPtr, const GenContext &genContext);

    ValueOrLogicalResult mlirGenCreateInterfaceVTableForClass(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              const GenContext &genContext);

    ValueOrLogicalResult mlirGenCreateInterfaceVTableForObject(mlir::Location location, mlir::Value in, 
            mlir_ts::ObjectType objectType, InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext);

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
                                                                        const GenContext &genContext);

    mlir::LogicalResult mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location,
                                                                       ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr,
                                                                       const GenContext &genContext);

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
            funcGenContext.thisClassType = newClassPtr->classType;
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
            auto &methodInfos = newClassPtr->methods;
            methodInfos.push_back(
            {
                fullClassStaticName.str(),
                funcType,
                fullClassStaticName.str(),
                true,
                false, 
                false, 
                -1, 
                posIndex,
                mlir_ts::AccessLevel::Public
            });
        }        

        return fullClassStaticName.str();
    }

    mlir::LogicalResult mlirGenClassBaseInterfaces(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                   const GenContext &genContext);

    mlir::LogicalResult mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             HeritageClause heritageClause,
                                                             const GenContext &genContext);

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrFieldInfo> &virtualTable)
    {
        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isField)
            {
                fields.push_back(
                {
                    vtableRecord.fieldInfo.id, 
                    mlir_ts::RefType::get(vtableRecord.fieldInfo.type), 
                    false, 
                    mlir_ts::AccessLevel::Public
                });
            }
            else
            {
                fields.push_back(
                {
                    MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()),
                    vtableRecord.methodInfo.funcType,
                    false,
                    mlir_ts::AccessLevel::Public
                });
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
                fields.push_back(
                {
                    MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()), 
                    getOpaqueType(),
                    false,
                    mlir_ts::AccessLevel::Public
                });
            }
            else
            {
                if (!vtableRecord.isStaticField)
                {
                    fields.push_back(
                    {
                        MLIRHelper::TupleFieldName(vtableRecord.methodInfo.name, builder.getContext()),
                        vtableRecord.methodInfo.funcType,
                        false,
                        mlir_ts::AccessLevel::Public
                    });
                }
                else
                {
                    fields.push_back(
                    {
                        vtableRecord.staticFieldInfo.id, 
                        mlir_ts::RefType::get(vtableRecord.staticFieldInfo.type),
                        false,
                        mlir_ts::AccessLevel::Public
                    });
                }
            }
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinition(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                           const GenContext &genContext);

    struct ClassMethodMemberInfo
    {
        ClassMethodMemberInfo(ClassInfo::TypePtr newClassPtr, ClassElement classMember) : newClassPtr(newClassPtr), classMember(classMember)
        {
            isConstructor = classMember == SyntaxKind::Constructor;
            isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
            isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
            auto isPrivate = hasModifier(classMember, SyntaxKind::PrivateKeyword);
            auto isProtected = hasModifier(classMember, SyntaxKind::ProtectedKeyword);
            //auto isPublic = hasModifier(classMember, SyntaxKind::PublicKeyword);

            accessLevel = mlir_ts::AccessLevel::Public;
            if (isPrivate)
            {
                accessLevel = mlir_ts::AccessLevel::Private;
            }
            else if (isProtected)
            {
                accessLevel = mlir_ts::AccessLevel::Protected;
            }

            isExport = newClassPtr->isExport && (isConstructor || accessLevel == mlir_ts::AccessLevel::Public);
            isImport = newClassPtr->isImport && (isConstructor || accessLevel == mlir_ts::AccessLevel::Public);
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

        mlir_ts::AccessLevel getAccessLevel()
        {
            return accessLevel;
        }

        bool registerClassMethodMember(mlir::Location location, int orderWeight, mlir_ts::AccessLevel accessLevel)
        {
            auto &methodInfos = newClassPtr->methods;

            auto methodIndex = newClassPtr->getMethodIndex(methodName);
            if (methodIndex < 0)
            {
                methodInfos.push_back(
                {
                    methodName,
                    getFuncType(),
                    getFuncName().str(),
                    isStatic,
                    isAbstract || isVirtual, 
                    isAbstract, 
                    -1, 
                    orderWeight, 
                    accessLevel
                });
            }
            else
            {
                methodInfos[methodIndex].orderWeight = orderWeight;
                methodInfos[methodIndex].accessLevel = accessLevel;
            }

            if (propertyName.size() > 0)
            {
                addAccessor(accessLevel);
            }

            if (newClassPtr->indexes.size() > 0)
            {
                if (methodName == INDEX_ACCESS_GET_FIELD_NAME)
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

                    indexer.get = makeFunctionEntry(funcOp);
                    indexer.getAccessLevel = accessLevel;
                }
                else if (methodName == INDEX_ACCESS_SET_FIELD_NAME)
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

                    indexer.set = makeFunctionEntry(funcOp);
                    indexer.setAccessLevel = accessLevel;
                }
            }

            return true;
        }

        void addAccessor(mlir_ts::AccessLevel accessLevel)
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
                newClassPtr->accessors[accessorIndex].get = makeFunctionEntry(funcOp);
                newClassPtr->accessors[accessorIndex].getAccessLevel = accessLevel;
            }
            else if (classMember == SyntaxKind::SetAccessor)
            {
                newClassPtr->accessors[accessorIndex].set = makeFunctionEntry(funcOp);
                newClassPtr->accessors[accessorIndex].setAccessLevel = accessLevel;
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
        mlir_ts::AccessLevel accessLevel;

        mlir_ts::FuncOp funcOp;
    };

    mlir::LogicalResult mlirGenClassIndexMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                const GenContext &genContext);

    mlir::LogicalResult mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 int orderWeight,
                                                 const GenContext &genContext);

    mlir::LogicalResult mlirGenClassStaticBlockMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 const GenContext &genContext);

    mlir::LogicalResult registerGenericClassMethod(ClassMethodMemberInfo &classMethodMemberInfo, const GenContext &genContext);

    mlir::LogicalResult mlirGenClassMethodMemberDynamicImport(ClassMethodMemberInfo &classMethodMemberInfo, int orderWeight, const GenContext &genContext);

    mlir::LogicalResult createGlobalConstructor(ClassElement classMember, const GenContext &genContext);

    mlir::LogicalResult generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, bool staticConstructor,
                                                      const GenContext &genContext);

    bool isConstValue(Expression expr, const GenContext &genContext)
    {
        auto isConst = false;
        evaluate(
            expr, [&](mlir::Value val) { isConst = isConstValue(val); }, genContext);
        return isConst;
    }

    mlir::LogicalResult registerGenericInterface(InterfaceDeclaration interfaceDeclarationAST,
                                                 const GenContext &genContext);

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
                                                const GenContext &genContext);

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(const std::string &name, bool &declareInterface,
                                                const GenContext &genContext);

    mlir::LogicalResult mlirGenInterfaceHeritageClauseExtends(InterfaceDeclaration interfaceDeclarationAST,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              HeritageClause heritageClause, int &orderWeight, bool declareClass,
                                                              const GenContext &genContext);

    mlir::LogicalResult mlirGen(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext);

    template <typename T> mlir::LogicalResult mlirGenInterfaceType(T newInterfacePtr, const GenContext &genContext)
    {
        if (newInterfacePtr)
        {
            newInterfacePtr->interfaceType = getInterfaceType(newInterfacePtr->fullName);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenInterfaceAddFieldMember(InterfaceInfo::TypePtr newInterfacePtr, mlir::Attribute fieldId, mlir::Type typeIn, bool isConditional, int orderWeight, bool declareInterface = true);

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
                                                     const GenContext &genContext);

    std::tuple<std::string, bool> getNameForMethod(SignatureDeclarationBase methodSignature, const GenContext &genContext);

    mlir::LogicalResult getMethodNameOrPropertyName(bool isStaticClass, SignatureDeclarationBase methodSignature, std::string &methodName,
                                                    std::string &propertyName, const GenContext &genContext);

    // RAII scope that redirects theModule and the builder into a fresh throwaway
    // module for the discovery pass. On scope exit the discovery module is erased
    // with everything the pass created, and theModule/builder are restored, so
    // discovery cleanup is structurally unable to touch real module content.
    class DiscoveryModuleScope
    {
      public:
        DiscoveryModuleScope(MLIRGenImpl &mlirGenImpl)
            : moduleGuard(mlirGenImpl.theModule), insertGuard(mlirGenImpl.builder)
        {
            discoveryModule =
                mlir::ModuleOp::create(mlirGenImpl.theModule.getLoc(), mlir::StringRef("discovery_module"));
            mlirGenImpl.theModule = discoveryModule;
            mlirGenImpl.builder.setInsertionPointToStart(discoveryModule.getBody());
        }

        ~DiscoveryModuleScope()
        {
            // members restore theModule and the insertion point after the erase
            discoveryModule.erase();
        }

      private:
        MLIRValueGuard<mlir::ModuleOp> moduleGuard;
        mlir::OpBuilder::InsertionGuard insertGuard;
        mlir::ModuleOp discoveryModule;
    };

    // RAII scope that redirects theModule and the builder into the temp module
    // for speculative evaluation and restores both when it goes out of scope.
    class TempModuleScope
    {
      public:
        TempModuleScope(MLIRGenImpl &mlirGenImpl)
            : moduleGuard(mlirGenImpl.theModule), insertGuard(mlirGenImpl.builder)
        {
            mlirGenImpl.builder.setInsertionPointToStart(mlirGenImpl.prepareTempModule());
        }

      private:
        MLIRValueGuard<mlir::ModuleOp> moduleGuard;
        mlir::OpBuilder::InsertionGuard insertGuard;
    };

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

        TempModuleScope tempModuleScope(*this);
        SymbolTableScopeT varScope(symbolTable);

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

    mlir::Value evaluatePropertyValue(mlir::Location location, mlir::Value exprValue, const std::string &propertyName, const GenContext &genContext)
    {
        // we need to ignore errors;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
        });

        TempModuleScope tempModuleScope(*this);

        GenContext evalGenContext(genContext);
        evalGenContext.allowPartialResolve = true;
        evalGenContext.funcOp = tempFuncOp;
        auto result = mlirGenPropertyAccessExpression(location, exprValue, propertyName, evalGenContext);
        return V(result);
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

        TempModuleScope tempModuleScope(*this);

        GenContext evalGenContext(genContext);
        evalGenContext.allowPartialResolve = true;
        auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getStructIndexType(),
                                mth.getStructIndexAttrValue(0));
        auto result = mlirGenElementAccess(location, expression, indexVal, isConditionalAccess, evalGenContext);
        auto initValue = V(result);
        return initValue ? initValue.getType() : mlir::Type();
    }

    ValueOrLogicalResult selectFieldsValues(mlir::Location location, SmallVector<mlir::Value> &values, mlir::Value value,  
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, bool filterSpecialCases, const GenContext &genContext, bool errorAsWarning = false);

    // TODO: needs to unified with selectFieldsValues
    ValueOrLogicalResult mapTupleToFields(mlir::Location location, SmallVector<mlir::Value> &values, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, bool filterSpecialCases, const GenContext &genContext, bool errorAsWarning = false);


    ValueOrLogicalResult castTupleToTuple(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ArrayRef<mlir_ts::FieldInfo> fields, const GenContext &genContext, bool errorAsWarning = false);

    ValueOrLogicalResult NewClassInstanceWithSettingFields(mlir::Location location, mlir_ts::ClassType classType, 
        ArrayRef<mlir_ts::FieldInfo> fields, ArrayRef<mlir::Value> values, const GenContext &genContext);

    ValueOrLogicalResult castTupleToClass(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ArrayRef<mlir_ts::FieldInfo> fields, mlir_ts::ClassType classType, const GenContext &genContext, bool errorAsWarning = false);

    ValueOrLogicalResult castFieldsToClass(mlir::Location location, mlir::Value value, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, 
        mlir_ts::ClassType classType, const GenContext &genContext, bool errorAsWarning = false);

    // TODO: finish it
    ValueOrLogicalResult castConstArrayToString(mlir::Location location, mlir::Value value, const GenContext &genContext);

    ValueOrLogicalResult castTupleToString(mlir::Location location, mlir::Value value, mlir_ts::TupleType tupleType,
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext);

    ValueOrLogicalResult generatingStaticNewCtorForClass(mlir::Location location, ClassInfo::TypePtr classInfo, int posIndex, const GenContext &genContext);

    ValueOrLogicalResult castClassToTuple(mlir::Location location, mlir::Value value, mlir_ts::ClassType classType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext);

    ValueOrLogicalResult castInterfaceToTuple(mlir::Location location, mlir::Value value, mlir_ts::InterfaceType interfaceType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext);

    // TODO: cast should not throw error in case of generic methods in "if (false)" conditions (typeof == "..."), 
    // as it may prevent cmpiling code
    ValueOrLogicalResult cast(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext, bool disableStrictNullCheck = false);

    // cast() stages; each returns a value/failure when the case is handled or std::nullopt to continue the cast pipeline

    mlir::LogicalResult verifyCastPreconditions(mlir::Location location, mlir::Type type, mlir::Type valueType, bool disableStrictNullCheck);

    std::optional<ValueOrLogicalResult> castViaToPrimitive(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // class or array or tuple to string
    std::optional<ValueOrLogicalResult> castToStringSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // class or tuple or object to interface
    std::optional<ValueOrLogicalResult> castToInterfaceSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // casts between tuple-like types (tuple, const tuple, class storage, interface fields)
    std::optional<ValueOrLogicalResult> castTupleLikeVariants(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // optional
    // TODO: it is in CastLogic as well, review usage and remove from here
    // but if optional points to interface then it will not work
    // example: from path.ts
    // %6 = ts.Cast %4 : !ts.const_tuple<{"key",!ts.string},{"prev",!ts.undefined},{"typename",!ts.undefined}> to !ts.optional<!ts.iface<@Path>>
    std::optional<ValueOrLogicalResult> castToOptionalType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    std::optional<ValueOrLogicalResult> castToTaggedUnionType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // union or optional or any or opaque source type
    std::optional<ValueOrLogicalResult> castFromSourceSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    mlir::LogicalResult verifyFunctionCastRules(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext);

    // cast ext method to bound method
    std::optional<ValueOrLogicalResult> castExtensionFunctionType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType);

    // wrong casts
    // TODO: put it into Cast::Verify
    mlir::LogicalResult verifyCastCompatibility(mlir::Location location, mlir::Type type, mlir::Type valueType);

    ValueOrLogicalResult castPrimitiveTypeFromAny(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext);

    // TODO: remove using typeof for Union types as it can't handle types such as 2 tuples in union etc
    ValueOrLogicalResult castFromUnion(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext);

    ValueOrLogicalResult castTupleToInterface(mlir::Location location, mlir::Value in, mlir::Type tupleTypeIn,
                                     mlir_ts::InterfaceType interfaceType, const GenContext &genContext);

    ValueOrLogicalResult castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    mlir_ts::InterfaceType interfaceType, const GenContext &genContext);

    ValueOrLogicalResult castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    InterfaceInfo::TypePtr interfaceInfo, const GenContext &genContext);

    mlir_ts::CreateBoundFunctionOp createBoundMethodFromExtensionMethod(mlir::Location location, mlir_ts::CreateExtensionFunctionOp createExtentionFunction);

    mlir::Type getType(Node typeReferenceAST, const GenContext &genContext);

    mlir::Type getInferType(mlir::Location location, InferTypeNode inferTypeNodeAST, const GenContext &genContext);

    mlir::Type getResolveTypeParameter(StringRef typeParamName, bool defaultType, const GenContext &genContext);

    mlir::Type getResolveTypeParameter(TypeParameterDeclaration typeParameterDeclaration, const GenContext &genContext);

    mlir::Type getTypeByTypeName(Node node, const GenContext &genContext);

    mlir::Type getFirstTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext);

    mlir::Type getSecondTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext);

    Reason testConstraint(mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, const GenContext &genContext);

    std::tuple<mlir::LogicalResult, IsGeneric> zipTypeParameterWithArgument(
        mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, bool noExtendTest,
        const GenContext &genContext, bool mergeTypes = false, bool arrayMerge = false);

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, llvm::ArrayRef<mlir::Type> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext);


    std::tuple<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext);

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithArgumentsNoDefaults(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext);

    std::pair<mlir::LogicalResult, IsGeneric> zipTypeParametersWithDefaultArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext);

    mlir::Type createTypeReferenceType(TypeReferenceNode typeReferenceAST, const GenContext &genContext);

    mlir::Type getTypeByTypeReference(mlir::Location location, mlir_ts::TypeReferenceType typeReferenceType, const GenContext &genContext);

    mlir::Type resolveGenericTypeInNamespace(mlir::Location location, StringRef name, TypeReferenceNode typeReferenceAST, const GenContext &genContext);

    mlir::Type resolveGenericType(mlir::Location location, StringRef name, TypeReferenceNode typeReferenceAST, const GenContext &genContext);

    mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext);

    mlir::Type findEmbeddedType(mlir::Location location, std::string name, NodeArray<TypeNode> &typeArguments, const GenContext &genContext);

    bool isEmbededType(mlir::StringRef name);
    
    bool isEmbededTypeWithBuiltins(mlir::StringRef name);

    bool isEmbededTypeWithNoBuiltins(mlir::StringRef name);

    mlir::Type getEmbeddedType(mlir::StringRef name);

    mlir::Type getEmbeddedTypeBuiltins(mlir::StringRef name);

    mlir::Type getEmbeddedTypeNoBuiltins(mlir::StringRef name);

    mlir::Type getEmbeddedTypeWithParam(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext);

    mlir::Type getEmbeddedTypeWithParamBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext);

    mlir::Type getEmbeddedTypeWithParamNoBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext);

    mlir::Type getEmbeddedTypeWithManyParams(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext);

    mlir::Type getEmbeddedTypeWithManyParamsBuiltins(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext);

    mlir::Type StringLiteralTypeFunc(mlir::Type type, std::function<std::string(StringRef)> f);

    mlir::Type UppercaseType(mlir::Type type);

    mlir::Type LowercaseType(mlir::Type type);

    mlir::Type CapitalizeType(mlir::Type type);

    mlir::Type UncapitalizeType(mlir::Type type);

    mlir::Type NonNullableTypes(mlir::Type type);

    // TODO: remove using those types as there issue with generic types
    mlir::Type ExcludeTypes(mlir::Location location, mlir::Type type, mlir::Type exclude);

    mlir::Type ExtractTypes(mlir::Location location, mlir::Type type, mlir::Type extract);

    mlir::Type RecordType(mlir::Type keys, mlir::Type valueType);

    mlir::Type PickTypes(mlir::Type type, mlir::Type keys);

    mlir::Type OmitTypes(mlir::Type type, mlir::Type keys);

    mlir::Type getTypeByTypeQuery(TypeQueryNode typeQueryAST, const GenContext &genContext);

    mlir::Type getTypePredicateType(TypePredicateNode typePredicateNode, const GenContext &genContext);

    // mutates genContext.typeParamsWithArgs with types inferred while resolving the conditional type
    mlir::Type processConditionalForType(ConditionalTypeNode conditionalTypeNode, mlir::Type checkType, mlir::Type extendsType, mlir::Type inferType, GenContext &genContext);

    mlir::Type getConditionalType(ConditionalTypeNode conditionalTypeNode, const GenContext &genContext);

    mlir::Type getKeyOf(TypeOperatorNode typeOperatorNode, const GenContext &genContext);

    mlir::Type getKeyOf(mlir::Location location, mlir::Type type, const GenContext &genContext);

    mlir::Type getTypeOperator(TypeOperatorNode typeOperatorNode, const GenContext &genContext);

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::ArrayType type);

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::ConstArrayType type);

    mlir::Type getIndexedAccessTypeForArrayElement(mlir_ts::StringType type);

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
    mlir::Type getIndexedAccessType(mlir::Type type, mlir::Type indexType, const GenContext &genContext);

    mlir::Type getIndexedAccessType(IndexedAccessTypeNode indexedAccessTypeNode, const GenContext &genContext);

    mlir::Type getTemplateLiteralType(TemplateLiteralTypeNode templateLiteralTypeNode, const GenContext &genContext);

    void getTemplateLiteralSpan(SmallVector<mlir::Type> &types, const std::string &head,
                                NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex, const GenContext &genContext);

    void getTemplateLiteralTypeItem(SmallVector<mlir::Type> &types, mlir_ts::LiteralType literalType, const std::string &head,
                                    NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                    const GenContext &genContext);

    void getTemplateLiteralUnionType(SmallVector<mlir::Type> &types, mlir::Type unionType, const std::string &head,
                                     NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                     const GenContext &genContext);

    mlir::Type getMappedType(MappedTypeNode mappedTypeNode, const GenContext &genContext);

    mlir_ts::VoidType getVoidType();

    mlir_ts::ByteType getByteType();

    mlir_ts::BooleanType getBooleanType();

    mlir_ts::NumberType getNumberType();

    mlir_ts::BigIntType getBigIntType();

    mlir::IndexType getIndexType();

    mlir_ts::StringType getStringType();

    mlir_ts::CharType getCharType();

    mlir_ts::EnumType getEnumType();

    mlir_ts::EnumType getEnumType(mlir::FlatSymbolRefAttr name, mlir::Type elementType, mlir::DictionaryAttr values);

    mlir_ts::ObjectStorageType getObjectStorageType(mlir::FlatSymbolRefAttr name);

    mlir_ts::ClassStorageType getClassStorageType(mlir::FlatSymbolRefAttr name);

    mlir_ts::ClassType getClassType(mlir::FlatSymbolRefAttr name, mlir::Type storageType);

    mlir_ts::NamespaceType getNamespaceType(mlir::StringRef name);

    mlir_ts::InterfaceType getInterfaceType(StringRef fullName);

    mlir_ts::InterfaceType getInterfaceType(mlir::FlatSymbolRefAttr name);

    mlir::Type getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size, const GenContext &genContext);

    mlir::Type getConstArrayType(mlir::Type elementType, unsigned size);

    mlir::Type getArrayType(ArrayTypeNode arrayTypeAST, const GenContext &genContext);

    mlir::Type getArrayType(mlir::Type elementType);

    mlir::Type getValueRefType(mlir::Type elementType);

    mlir_ts::NamedGenericType getNamedGenericType(StringRef name);

    mlir_ts::InferType getInferType(mlir::Type paramType);

    mlir::Type getConditionalType(mlir::Type checkType, mlir::Type extendsType, mlir::Type trueType, mlir::Type falseType);

    mlir::Type getIndexAccessType(mlir::Type index, mlir::Type indexAccess);

    mlir::Type getKeyOfType(mlir::Type type);

    mlir::Type getMappedType(mlir::Type elementType, mlir::Type nameType, mlir::Type constrainType);

    mlir_ts::TypeReferenceType getTypeReferenceType(mlir::StringRef nameRef, mlir::SmallVector<mlir::Type> &types);

    mlir::Value getUndefined(mlir::Location location);

    mlir::Value getInfinity(mlir::Location location);

    mlir::Value getNaN(mlir::Location location);

    std::pair<mlir::Attribute, mlir::LogicalResult> getNameFromComputedPropertyName(Node name, const GenContext &genContext);

    mlir::Attribute TupleFieldName(Node name, const GenContext &genContext);

    std::pair<bool, mlir::LogicalResult> getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext);

    mlir::LogicalResult getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext);

    mlir::Type getConstTupleType(TupleTypeNode tupleType, const GenContext &genContext);

    mlir_ts::ConstTupleType getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos);

    mlir::Type getTupleType(TupleTypeNode tupleType, const GenContext &genContext);

    mlir::Type getTupleType(TypeLiteralNode typeLiteral, const GenContext &genContext);

    mlir::Type getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos);

    mlir_ts::ObjectType getObjectType(mlir::Type type);

    mlir_ts::OpaqueType getOpaqueType();

    mlir_ts::BoundFunctionType getBoundFunctionType(mlir_ts::FunctionType funcType);

    mlir_ts::BoundFunctionType getBoundFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                                    bool isVarArg);

    mlir_ts::FunctionType getFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                          bool isVarArg);

    mlir_ts::ExtensionFunctionType getExtensionFunctionType(mlir_ts::FunctionType funcType);

    mlir::Type getSignature(SignatureDeclarationBase signature, const GenContext &genContext);

    mlir::Type getFunctionType(SignatureDeclarationBase signature, const GenContext &genContext);

    mlir::Type getConstructorType(SignatureDeclarationBase signature, const GenContext &genContext);

    mlir::Type getCallSignature(CallSignatureDeclaration signature, const GenContext &genContext);

    mlir::Type getConstructSignature(ConstructSignatureDeclaration constructSignature,
                                                const GenContext &genContext);

    mlir::Type getMethodSignature(MethodSignature methodSignature, const GenContext &genContext);

    mlir::Type getIndexSignature(IndexSignatureDeclaration indexSignature, const GenContext &genContext);

    mlir::Type getUnionType(UnionTypeNode unionTypeNode, const GenContext &genContext);

    mlir::Type getUnionType(mlir::Location location, mlir::Type type1, mlir::Type type2);

    mlir::Type getUnionType(mlir::SmallVector<mlir::Type> &types);

    mlir::LogicalResult processIntersectionType(InterfaceInfo::TypePtr newInterfaceInfo, mlir::Type type, bool conditional = false);

    mlir::Type getIntersectionType(IntersectionTypeNode intersectionTypeNode, const GenContext &genContext);

    mlir::Type getIntersectionType(mlir::Type type1, mlir::Type type2);

    mlir::Type getIntersectionType(mlir::SmallVector<mlir::Type> &types);

    mlir::Type AndType(mlir::Type left, mlir::Type right);

    mlir::Type AndUnionType(mlir_ts::UnionType leftUnion, mlir::Type right);

    InterfaceInfo::TypePtr newInterfaceType(IntersectionTypeNode intersectionTypeNode, bool &declareInterface,
                                            const GenContext &genContext);

    mlir::LogicalResult mergeInterfaces(InterfaceInfo::TypePtr dest, mlir_ts::TupleType src, bool conditional = false);

    mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode, const GenContext &genContext);

    mlir::Type getLiteralType(LiteralTypeNode literalTypeNode);

    mlir::Type getOptionalType(OptionalTypeNode optionalTypeNode, const GenContext &genContext);

    mlir::Type getOptionalType(mlir::Type type);

    mlir::Type getRestType(RestTypeNode restTypeNode, const GenContext &genContext);

    mlir_ts::AnyType getAnyType();

    mlir_ts::UnknownType getUnknownType();

    mlir_ts::NeverType getNeverType();

    mlir_ts::ConstType getConstType();

    mlir_ts::SymbolType getSymbolType();

    mlir_ts::UndefinedType getUndefinedType();

    mlir_ts::NullType getNullType();

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
            if (previousVariable && previousVariable.getParentBlock() == value.getParentBlock())
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

    void iterateDecorators(Node node, const GenContext &genContext, std::function<void(StringRef, SmallVector<StringRef>)> functor)
    {
        for (auto decorator : node->modifiers)
        {
            if (decorator != SyntaxKind::Decorator)
            {
                continue;
            }

            SmallVector<StringRef> args;
            auto expr = decorator.as<Decorator>()->expression;
            if (expr == SyntaxKind::CallExpression)
            {
                auto callExpression = expr.as<CallExpression>();
                expr = callExpression->expression;
                for (auto argExpr : callExpression->arguments)
                {
                    if (argExpr == SyntaxKind::NumericLiteral)
                    {
                        auto num = argExpr.as<NumericLiteral>();
                        args.push_back(mlir::StringRef(convertWideToUTF8(num->text)).copy(stringAllocator));
                        continue;
                    }

                    if (argExpr == SyntaxKind::StringLiteral)
                    {
                        args.push_back(MLIRHelper::getName(argExpr.as<Node>(), stringAllocator));
                        continue;
                    }

                    auto resultType = evaluate(argExpr, genContext);
                    if (auto litType = dyn_cast<mlir_ts::LiteralType>(resultType))
                    {
                        mlir::Attribute value = litType.getValue();
                        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value)) 
                        {
                            auto val = llvm::toString(intAttr.getValue(), 10, false);
                            args.push_back(mlir::StringRef(val).copy(stringAllocator));
                        }
                        else if (auto strAttr = dyn_cast<mlir::StringAttr>(value)) 
                        {
                            args.push_back(strAttr.getValue());
                        }

                        continue;
                    }

                    // TODO: finish it
                }
            }            

            if (expr == SyntaxKind::Identifier)
            {
                auto name = MLIRHelper::getName(expr.as<Node>(), stringAllocator);
                functor(name, args);
            }
        }
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
                if (!interfaceInfo)
                {
                    // not registered (e.g. forward/ambient reference), nothing to export
                    return true;
                }

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
                if (!classInfo)
                {
                    // not registered (e.g. forward/ambient reference), nothing to export
                    return true;
                }

                for (auto& method : classInfo->methods)
                {
                    addDependancyTypesToExport(method.funcType);
                }

                for (auto& accessor : classInfo->accessors)
                {
                    if (accessor.get) addDependancyTypesToExport(accessor.get.funcType);
                    if (accessor.set) addDependancyTypesToExport(accessor.set.funcType);
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
                if (!interfaceInfo)
                {
                    // not registered (e.g. forward/ambient reference), nothing to export
                    return true;
                }

                addInterfaceDeclarationToExport(interfaceInfo);
                return true;
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                if (!classInfo)
                {
                    // not registered (e.g. forward/ambient reference), nothing to export
                    return true;
                }

                addClassDeclarationToExport(classInfo);
                return true;
            })
            .Case<mlir_ts::EnumType>([&](auto enumType) {
                auto enumInfo = getEnumInfoByFullName(enumType.getName().getValue());
                if (!enumInfo || enumInfo->enumType != enumType)
                {
                    // not registered (e.g. forward/ambient reference), nothing to export
                    return true;
                }

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

    void addGenericFunctionDeclarationToExport(GenericFunctionInfo::TypePtr genericFunctionInfo)
    {
        // funcType can be null when registered with ignoreFunctionArgsDetection=true
        // (see registerGenericFunctionLike) - nothing to key isAddedToExport's dedup on
        // in that case, so just fall through and re-print (matches
        // addFunctionDeclarationToExport's own "TODO: add distinct declaration" gap for
        // the non-generic case).
        if (genericFunctionInfo->funcType && isAddedToExport(genericFunctionInfo->funcType))
        {
            // already added
            return;
        }

        // same bounds-check as addGenericClassDeclarationToExport - a generic function
        // re-declared while re-importing another module's embedded declarations still
        // carries its own `export` keyword verbatim, but at that point `sourceFile` is
        // the ambient/outer file, not the "partial" buffer parsePartialStatements
        // actually parsed this declaration from.
        auto declEnd = static_cast<size_t>(genericFunctionInfo->functionDeclaration->_end);
        if (declEnd > genericFunctionInfo->sourceFile->text.length())
        {
            return;
        }

        if (genericFunctionInfo->funcType)
        {
            exportedTypes.insert(genericFunctionInfo->funcType);
        }

        // like a generic class, a generic function has no compiled body for any given
        // instantiation: each importing module instantiates it locally, on demand,
        // exactly like a same-file usage would. So the FULL original source - type
        // parameters and body intact, no @dllimport marker - must be re-exported
        // verbatim for parsePartialStatements to recompile per instantiation in the
        // importer.
        auto declText = convertWideToUTF8(getTextOfNodeFromSourceText(
            genericFunctionInfo->sourceFile->text, genericFunctionInfo->functionDeclaration.as<Node>(), true));

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);
        MLIRDeclarationPrinter dp(ss);
        dp.printGenericClass(genericFunctionInfo->elementNamespace, declText);

        genericDeclExports << ss.str().str();
    }

    void addGenericClassDeclarationToExport(GenericClassInfo::TypePtr genericClassInfo)
    {
        if (isAddedToExport(genericClassInfo->classType))
        {
            // already added
            return;
        }

        // A generic class re-declared while re-importing another module's embedded
        // declarations (see mlirGenImportSharedLib) still carries its own `export` keyword
        // verbatim (it is the original source, copied as-is), so it reaches this same
        // function a second time from inside the IMPORTER's own compile - but at that
        // point `sourceFile` is the ambient/outer file (e.g. the importer's own .ts), not
        // the "partial" buffer parsePartialStatements actually parsed this declaration
        // from, so classDeclaration's pos/_end (offsets into THAT buffer) do not correspond
        // to sourceFile->text at all and can exceed its length. Re-exporting a
        // re-declaration one level removed like this is also simply wrong (the importer
        // isn't meant to re-export M's generics further) - bounds-check and skip rather
        // than let getTextOfNodeFromSourceText's substr throw std::out_of_range.
        auto declEnd = static_cast<size_t>(genericClassInfo->classDeclaration->_end);
        if (declEnd > genericClassInfo->sourceFile->text.length())
        {
            return;
        }

        exportedTypes.insert(genericClassInfo->classType);

        // unlike a concrete class (printed above as a signature-only, @dllimport-marked
        // declaration - the real body is already compiled into this module's DLL), a
        // generic class has no compiled body for any given instantiation: each importing
        // module instantiates it locally, on demand, exactly like a same-file usage would.
        // So the FULL original source - type parameters and method bodies intact, no
        // @dllimport marker - must be re-exported verbatim for parsePartialStatements to
        // recompile per instantiation in the importer.
        auto declText = convertWideToUTF8(getTextOfNodeFromSourceText(
            genericClassInfo->sourceFile->text, genericClassInfo->classDeclaration.as<Node>(), true));

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);
        MLIRDeclarationPrinter dp(ss);
        dp.printGenericClass(genericClassInfo->elementNamespace, declText);

        genericDeclExports << ss.str().str();
    }

    void addGenericInterfaceDeclarationToExport(GenericInterfaceInfo::TypePtr genericInterfaceInfo)
    {
        if (genericInterfaceInfo->interfaceType && isAddedToExport(genericInterfaceInfo->interfaceType))
        {
            // already added
            return;
        }

        // same bounds-check as addGenericClassDeclarationToExport - a generic interface
        // re-declared while re-importing another module's embedded declarations still
        // carries its own `export` keyword verbatim, but at that point `sourceFile` is
        // the ambient/outer file, not the "partial" buffer parsePartialStatements
        // actually parsed this declaration from.
        auto declEnd = static_cast<size_t>(genericInterfaceInfo->interfaceDeclaration->_end);
        if (declEnd > genericInterfaceInfo->sourceFile->text.length())
        {
            return;
        }

        if (genericInterfaceInfo->interfaceType)
        {
            exportedTypes.insert(genericInterfaceInfo->interfaceType);
        }

        // like a generic class, a generic interface has no compiled body for any given
        // instantiation: each importing module instantiates it locally, on demand,
        // exactly like a same-file usage would. So the FULL original source - type
        // parameters and member signatures intact, no @dllimport marker - must be
        // re-exported verbatim for parsePartialStatements to recompile per instantiation
        // in the importer.
        auto declText = convertWideToUTF8(getTextOfNodeFromSourceText(
            genericInterfaceInfo->sourceFile->text, genericInterfaceInfo->interfaceDeclaration.as<Node>(), true));

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);
        MLIRDeclarationPrinter dp(ss);
        dp.printGenericClass(genericInterfaceInfo->elementNamespace, declText);

        genericDeclExports << ss.str().str();
    }

    void addGenericTypeAliasDeclarationToExport(TypeAliasDeclaration typeAliasDeclarationAST, NamespaceInfo::TypePtr elementNamespace)
    {
        // same bounds-check as addGenericClassDeclarationToExport - a generic type alias
        // re-declared while re-importing another module's embedded declarations still
        // carries its own `export` keyword verbatim, but at that point `sourceFile` is
        // the ambient/outer file, not the "partial" buffer parsePartialStatements
        // actually parsed this declaration from.
        auto declEnd = static_cast<size_t>(typeAliasDeclarationAST->_end);
        if (declEnd > sourceFile->text.length())
        {
            return;
        }

        // like a generic class, a generic type alias has no compiled body for any given
        // instantiation: resolution is pure compile-time type substitution, done fresh in
        // whichever module references it. So the FULL original source - type parameters
        // and the aliased type expression intact, no @dllimport marker - must be
        // re-exported verbatim for parsePartialStatements to recompile per reference in
        // the importer.
        auto declText = convertWideToUTF8(getTextOfNodeFromSourceText(
            sourceFile->text, typeAliasDeclarationAST.as<Node>(), true));

        SmallVector<char> out;
        llvm::raw_svector_ostream ss(out);
        MLIRDeclarationPrinter dp(ss);
        dp.printGenericClass(elementNamespace, declText);

        genericDeclExports << ss.str().str();
    }

    auto getNamespaceName() -> StringRef
    {
        return currentNamespace->name;
    }

    auto getFullNamespaceName() -> StringRef
    {
        return currentNamespace->fullName;
    }

    // no interning - use for lookups and transient names; getFullNamespaceName interns for names that are stored
    auto concatFullNamespaceName(StringRef name) -> std::string
    {
        if (currentNamespace->fullName.empty())
        {
            return name.str();
        }

        std::string res;
        res.reserve(currentNamespace->fullName.size() + name.size() + 1);
        res += currentNamespace->fullName;
        res += ".";
        res += name;
        return res;
    }

    auto getFullNamespaceName(StringRef name) -> StringRef
    {
        return StringRef(concatFullNamespaceName(name)).copy(stringAllocator);
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

    auto getFunctionMap() -> llvm::StringMap<FunctionEntry> &
    {
        return currentNamespace->functionMap;
    }

    auto lookupFunctionMap(StringRef name) -> FunctionEntry
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

    auto getTypeAliasMap() -> llvm::StringMap<std::pair<mlir::Type, TypeNode>> &
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

        // auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start + length - 1);
        // auto end = mlir::FileLineColLoc::get(builder.getContext(), fileId, 
        //     endLineChar.line + 1, endLineChar.character + 1);
        //return mlir::FusedLoc::get(builder.getContext(), {begin, end});
        return begin;
    }

    mlir::Location loc2Fuse(ts::SourceFile sourceFile, std::string fileName, int start, int length)
    {
        auto fileId = getStringAttr(fileName);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start);
        auto begin = mlir::FileLineColLoc::get(builder.getContext(), fileId, 
            posLineChar.line + 1, posLineChar.character + 1);
        // if (length <= 1)
        // {
        //     return begin;
        // }

        // auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, start + length - 1);
        // auto end = mlir::FileLineColLoc::get(builder.getContext(), 
        //     fileId, endLineChar.line + 1, endLineChar.character + 1);
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

    mlir::Location combine(mlir::Location parentLocation, mlir::Location location) 
    {
        if (isa<mlir::UnknownLoc>(parentLocation))
        {
            return location;
        }

        return mlir::FusedLoc::get(builder.getContext(), {parentLocation, location});  
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
        SmallString<128> exportType;
        raw_svector_ostream rso(exportType);        

        MLIRPrinter mp{};
        mp.printType<raw_svector_ostream>(rso, type);
        return exportType.str().str();      
    }

    void printDebug(ts::Node node)
    {
        Printer<std::wostream> printer(std::wcerr);
        printer.newLine();
        printer.printText("dump ===============================================");
        printer.newLine();
        printer.printNode(node);
        printer.newLine();
        printer.printText("end of dump ========================================");
        printer.newLine();
    }

    std::string print(ts::Node node)
    {
        sstream ss;
        Printer<sstream> printer(ss);
        printer.printNode(node);
        return convertWideToUTF8(ss.str());
    }    

    // TODO: fix issue with cercular reference of include files
    std::pair<SourceFile, std::vector<SourceFile>> loadIncludeFile(mlir::Location location, StringRef fileName)
    {
        SmallString<256> fileNameStr(fileName);

        if (fileNameStr.starts_with("./"))
        {
            auto subStr = fileNameStr.substr(2);
            fileNameStr.clear();
            fileNameStr.append(subStr);
        }

        if (sys::path::extension(fileName) == "")
        {
            fileNameStr += ".ts";
        }

        SmallString<256> fullPath;

        if (!sys::path::has_root_path(fileNameStr)) {
            // get dir from mainSourceFileName
            auto directory = sys::path::parent_path(mainSourceFileName);
            sys::path::append(fullPath, directory);
        }

        sys::path::append(fullPath, fileNameStr);

        std::string ignored;
        auto id = sourceMgr.AddIncludeFile(std::string(fullPath), SMLoc(), ignored);
        if (!id)
        {
            emitError(location, "can't open file: ") << fullPath;
            return {SourceFile(), {}};
        }

        const auto *sourceBuf = sourceMgr.getMemoryBuffer(id);
        auto sourceFileLoc = mlir::FileLineColLoc::get(builder.getContext(),
                    sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);        
        return loadSourceBuf(sourceFileLoc, sourceBuf);
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

    // Caches the stack-allocated ref materialized for a storage-less value (e.g. a
    // `const` binding with no backing storage) the first time a bound-method property
    // access needs an address for it (see MLIRPropertyAccessCodeLogic::TupleNoError).
    // Without this, each access re-materializes a fresh copy seeded from the pristine,
    // never-mutated SSA value, so repeated calls like `g.next()` on a `const`-bound
    // generator never observe state changes made by earlier calls. Keyed by mlir::Value
    // identity, which is stable and unique within a function. Scoped (not just cleared)
    // at each mlirGenFunctionBody entry via BoundRefCacheScopeT, mirroring symbolTable's
    // own scoping -- codegen for a nested closure recurses into mlirGenFunctionBody
    // while the enclosing function's generation is still on the call stack, so a plain
    // clear-on-entry would permanently drop the outer function's cache entries instead
    // of restoring them when the nested closure's generation finishes.
    llvm::ScopedHashTable<mlir::Value, mlir::Value> boundRefMaterializedCache;

    NamespaceInfo::TypePtr rootNamespace;

    NamespaceInfo::TypePtr currentNamespace;

    llvm::ScopedHashTable<StringRef, NamespaceInfo::TypePtr> fullNamespacesMap;

    llvm::ScopedHashTable<StringRef, GenericFunctionInfo::TypePtr> fullNameGenericFunctionsMap;

    llvm::ScopedHashTable<StringRef, EnumInfo::TypePtr> fullNameEnumsMap;

    llvm::ScopedHashTable<StringRef, ClassInfo::TypePtr> fullNameClassesMap;

    llvm::ScopedHashTable<StringRef, GenericClassInfo::TypePtr> fullNameGenericClassesMap;

    llvm::ScopedHashTable<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMap;

    // (object literal's ObjectStorageType -> (field name -> lifted function symbol)) for
    // capture-free-or-not object-literal methods; see addObjectFuncFieldInfo's doc comment
    // and docs/interface-vtable-simplification-design.md §3. Not namespace-scoped: an
    // ObjectStorageType's symbol already embeds the literal's source location, so it's
    // globally unique regardless of which namespace declared it.
    mlir::DenseMap<mlir::Type, mlir::DenseMap<mlir::Attribute, std::string>> objectLiteralMethodSymbolsMap;

    // fieldType is a method-as-field's FunctionType, whose first input is the object literal's
    // own "this" type (ObjectType wrapping the ObjectStorageType key used in
    // objectLiteralMethodSymbolsMap - see addObjectFuncFieldInfo). Returns the empty string if
    // fieldType isn't a function, "this" isn't an ObjectType (e.g. a class instance, or an
    // imported object type reconstructed from a @dllimport declaration with no local funcOp),
    // or no method was ever registered for this exact (object, field) pair.
    std::string lookupObjectLiteralMethodSymbol(mlir::Type fieldType, mlir::Attribute fieldId)
    {
        auto funcType = dyn_cast<mlir_ts::FunctionType>(fieldType);
        if (!funcType || funcType.getInputs().empty())
        {
            return {};
        }

        auto thisObjectType = dyn_cast<mlir_ts::ObjectType>(funcType.getInputs().front());
        if (!thisObjectType)
        {
            return {};
        }

        auto symbolsForObject = objectLiteralMethodSymbolsMap.find(thisObjectType.getStorageType());
        if (symbolsForObject == objectLiteralMethodSymbolsMap.end())
        {
            return {};
        }

        auto symbolIt = symbolsForObject->second.find(fieldId);
        return symbolIt == symbolsForObject->second.end() ? std::string{} : symbolIt->second;
    }

    llvm::ScopedHashTable<StringRef, GenericInterfaceInfo::TypePtr> fullNameGenericInterfacesMap;

    llvm::ScopedHashTable<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMap;

    llvm::ScopedHashTable<StringRef, mlir::LLVM::DIScopeAttr> debugScope;

    llvm::ScopedHashTable<SafeTypeKeyType, mlir::Value> safeTypesMap;

    // helper to get line number
    Parser parser;
    ts::SourceFile sourceFile;

    bool declarationMode;

    std::stringstream declExports;
    // generic class declarations must be re-imported as plain (non-".d.ts") source: unlike
    // declExports (parsed with a "partial.d.ts" filename, which makes the parser mark
    // everything ambient/external regardless of any per-declaration @dllimport marker - see
    // createDeclarationExportGlobalVar/mlirGenImportSharedLib), a generic class has no
    // compiled body for any instantiation to link against - the importer instantiates it
    // locally, and needs its real method bodies to actually be compilable, not treated as
    // external stubs. Kept in a separate stream/global so it can be re-parsed with a plain
    // ".ts" filename instead.
    std::stringstream genericDeclExports;
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
} // namespace mlirgen
} // namespace typescript
