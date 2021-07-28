#define DEBUG_TYPE "mlir"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "TypeScript/MLIRGenLogic.h"
#include "TypeScript/VisitorAST.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

#include <numeric>

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

// TODO: optimize of amount of calls to detect return types and if it is was calculated before then do not run it all the time

namespace
{

enum class VariableClass
{
    Const,
    Let,
    Var,
    ConstRef
};

struct StaticFieldInfo
{
    mlir::Attribute id;
    mlir::StringRef globalVariableName;
};

struct MethodInfo
{
    std::string name;
    mlir_ts::FuncOp funcOp;
    bool isStatic;
    bool isVirtual;
    int virtualIndex;
};

struct VirtualMethodOrInterfaceVTableInfo
{
    VirtualMethodOrInterfaceVTableInfo() = default;

    MethodInfo methodInfo;
    bool isInterfaceVTable;
};

struct AccessorInfo
{
    std::string name;
    mlir_ts::FuncOp get;
    mlir_ts::FuncOp set;
    bool isStatic;
    bool isVirtual;
};

struct InterfaceFieldInfo
{
    mlir::Attribute id;
    mlir::Type type;
};

struct InterfaceMethodInfo
{
    std::string name;
    mlir::FunctionType funcType;
    int virtualIndex;
};

struct InterfaceInfo
{
  public:
    using TypePtr = std::shared_ptr<InterfaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::InterfaceType interfaceType;

    llvm::SmallVector<InterfaceInfo::TypePtr> implements;

    llvm::SmallVector<InterfaceFieldInfo> fields;

    llvm::SmallVector<InterfaceMethodInfo> methods;

    InterfaceInfo()
    {
    }

    mlir::LogicalResult getVirtualTable(llvm::SmallVector<MethodInfo> &vtable,
                                        std::function<MethodInfo &(std::string, mlir::FunctionType)> resolveMethod)
    {
        // do vtable for current class
        for (auto &method : methods)
        {
            auto &classMethodInfo = resolveMethod(method.name, method.funcType);
            if (classMethodInfo.name.empty())
            {
                return mlir::failure();
            }

            vtable.push_back(classMethodInfo);
        }

        return mlir::success();
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(methods.begin(), std::find_if(methods.begin(), methods.end(),
                                                                [&](InterfaceMethodInfo methodInfo) { return name == methodInfo.name; }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    int getFieldIndex(mlir::Attribute id)
    {
        auto dist = std::distance(
            fields.begin(), std::find_if(fields.begin(), fields.end(), [&](InterfaceFieldInfo fieldInfo) { return id == fieldInfo.id; }));
        return (signed)dist >= (signed)fields.size() ? -1 : dist;
    }
};

struct ImplementInfo
{
    InterfaceInfo::TypePtr implement;
    int virtualIndex;
};

struct ClassInfo
{
  public:
    using TypePtr = std::shared_ptr<ClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir_ts::ClassType classType;

    llvm::SmallVector<ClassInfo::TypePtr> baseClasses;

    llvm::SmallVector<ImplementInfo> implements;

    llvm::SmallVector<StaticFieldInfo> staticFields;

    llvm::SmallVector<MethodInfo> methods;

    llvm::SmallVector<AccessorInfo> accessors;

    bool hasConstructor;
    bool hasInitializers;
    bool hasVirtualTable;
    bool isAbstract;

    ClassInfo() : hasConstructor(false), hasInitializers(false), hasVirtualTable(false), isAbstract(false)
    {
    }

    auto getHasConstructor() -> bool
    {
        if (hasConstructor)
        {
            return true;
        }

        for (auto &base : baseClasses)
        {
            if (base->hasConstructor)
            {
                return true;
            }
        }

        return false;
    }

    auto getHasVirtualTable() -> bool
    {
        if (hasVirtualTable)
        {
            return true;
        }

        for (auto &base : baseClasses)
        {
            if (base->hasVirtualTable)
            {
                return true;
            }
        }

        return false;
    }

    auto getHasVirtualTableVariable() -> bool
    {
        for (auto &base : baseClasses)
        {
            if (base->hasVirtualTable)
            {
                return false;
            }
        }

        if (hasVirtualTable)
        {
            return true;
        }

        return false;
    }

    void getVirtualTable(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &vtable)
    {
        for (auto &base : baseClasses)
        {
            base->getVirtualTable(vtable);
        }

        // do vtable for current class
        for (auto &implement : implements)
        {
            auto index = std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableRecord) {
                                           return implement.implement->fullName == vTableRecord.methodInfo.name;
                                       }));
            if ((size_t)index < vtable.size())
            {
                // found interface
                continue;
            }

            MethodInfo methodInfo;
            methodInfo.name = implement.implement->fullName.str();
            implement.virtualIndex = vtable.size();
            vtable.push_back({methodInfo, true});
        }

        // methods
        for (auto &method : methods)
        {
            auto index = std::distance(vtable.begin(), std::find_if(vtable.begin(), vtable.end(), [&](auto vTableMethod) {
                                           return method.name == vTableMethod.methodInfo.name;
                                       }));
            if ((size_t)index < vtable.size())
            {
                // found method
                vtable[index].methodInfo.funcOp = method.funcOp;
                method.isVirtual = true;
                continue;
            }

            if (method.isVirtual)
            {
                method.virtualIndex = vtable.size();
                vtable.push_back({method, false});
            }
        }
    }

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;

    int getStaticFieldIndex(mlir::Attribute id)
    {
        auto dist = std::distance(staticFields.begin(), std::find_if(staticFields.begin(), staticFields.end(),
                                                                     [&](StaticFieldInfo fldInf) { return id == fldInf.id; }));
        return (signed)dist >= (signed)staticFields.size() ? -1 : dist;
    }

    int getMethodIndex(mlir::StringRef name)
    {
        auto dist = std::distance(
            methods.begin(), std::find_if(methods.begin(), methods.end(), [&](MethodInfo methodInfo) { return name == methodInfo.name; }));
        return (signed)dist >= (signed)methods.size() ? -1 : dist;
    }

    int getAccessorIndex(mlir::StringRef name)
    {
        auto dist = std::distance(accessors.begin(), std::find_if(accessors.begin(), accessors.end(),
                                                                  [&](AccessorInfo accessorInfo) { return name == accessorInfo.name; }));
        return (signed)dist >= (signed)accessors.size() ? -1 : dist;
    }

    int getImplementIndex(mlir::StringRef name)
    {
        auto dist = std::distance(implements.begin(), std::find_if(implements.begin(), implements.end(), [&](ImplementInfo implementInfo) {
                                      return name == implementInfo.implement->fullName;
                                  }));
        return (signed)dist >= (signed)implements.size() ? -1 : dist;
    }
};

struct NamespaceInfo
{
  public:
    using TypePtr = std::shared_ptr<NamespaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    llvm::StringMap<mlir_ts::FuncOp> functionMap;

    llvm::StringMap<VariableDeclarationDOM::TypePtr> globalsMap;

    llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> captureVarsMap;

    llvm::StringMap<mlir::Type> typeAliasMap;

    llvm::StringMap<mlir::StringRef> importEqualsMap;

    llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> enumsMap;

    llvm::StringMap<ClassInfo::TypePtr> classesMap;

    llvm::StringMap<InterfaceInfo::TypePtr> interfacesMap;

    llvm::StringMap<NamespaceInfo::TypePtr> namespacesMap;
};

/// Implementation of a simple MLIR emission from the TypeScript AST.
///
/// This will emit operations that are specific to the TypeScript language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl
{
  public:
    MLIRGenImpl(const mlir::MLIRContext &context) : builder(&const_cast<mlir::MLIRContext &>(context))
    {
        fileName = "<unknown>";
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    MLIRGenImpl(const mlir::MLIRContext &context, const llvm::StringRef &fileNameParam) : builder(&const_cast<mlir::MLIRContext &>(context))
    {
        fileName = fileNameParam;
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    mlir::ModuleOp mlirGenSourceFile(SourceFile module)
    {
        sourceFile = module;

        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(loc(module), fileName);
        builder.setInsertionPointToStart(theModule.getBody());

        SymbolTableScopeT varScope(symbolTable);
        llvm::ScopedHashTableScope<StringRef, NamespaceInfo::TypePtr> fullNamespacesMapScope(fullNamespacesMap);
        llvm::ScopedHashTableScope<StringRef, ClassInfo::TypePtr> fullNameClassesMapScope(fullNameClassesMap);
        llvm::ScopedHashTableScope<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMapScope(fullNameInterfacesMap);
        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(fullNameGlobalsMap);

        // Process of discovery here
        GenContext genContextPartial = {0};
        genContextPartial.allowPartialResolve = true;
        genContextPartial.dummyRun = true;
        genContextPartial.cleanUps = new mlir::SmallVector<mlir::Block *>();
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            GenContext genContext = {0};
            for (auto &statement : module->statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContextPartial)))
                {
                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                }
            }

            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                theModule.emitError("can't resolve dependencies");
                return nullptr;
            }

        } while (notResolved > 0);

        genContextPartial.clean();

        // clean up
        theModule.getBody()->clear();

        // clear state
        for (auto &statement : module->statements)
        {
            statement->processed = false;
        }

        // Process generating here
        GenContext genContext = {0};
        for (auto &statement : module->statements)
        {
            if (failed(mlirGen(statement, genContext)))
            {
                return nullptr;
            }
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the TypeScript operations.
        if (failed(mlir::verify(theModule)))
        {
            // TODO: uncomment it
            theModule.emitError("module verification error");
            // return nullptr;
        }

        return theModule;
    }

    mlir::LogicalResult mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(moduleDeclarationAST);

        auto namespaceName = MLIRHelper::getName(moduleDeclarationAST->name);
        auto namePtr = StringRef(namespaceName).copy(stringAllocator);

        auto savedNamespace = currentNamespace;

        auto fullNamePtr = getFullNamespaceName(namePtr);
        auto &namespacesMap = getNamespaceMap();
        auto it = namespacesMap.find(namePtr);
        if (it == namespacesMap.end())
        {
            auto newNamespacePtr = std::make_shared<NamespaceInfo>();
            newNamespacePtr->name = namePtr;
            newNamespacePtr->fullName = fullNamePtr;
            namespacesMap.insert({namePtr, newNamespacePtr});
            fullNamespacesMap.insert(fullNamePtr, newNamespacePtr);
            currentNamespace = newNamespacePtr;
        }
        else
        {
            currentNamespace = it->getValue();
        }

        GenContext moduleGenContext = {0};
        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        currentNamespace = savedNamespace;

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        auto isNamespace = (moduleDeclarationAST->flags & NodeFlags::Namespace) == NodeFlags::Namespace;
        auto isNestedNamespace = (moduleDeclarationAST->flags & NodeFlags::NestedNamespace) == NodeFlags::NestedNamespace;
        if (isNamespace || isNestedNamespace)
        {
            return mlirGenNamespace(moduleDeclarationAST, genContext);
        }

        auto location = loc(moduleDeclarationAST);

        auto moduleName = MLIRHelper::getName(moduleDeclarationAST->name);

        auto moduleOp = builder.create<mlir::ModuleOp>(location, StringRef(moduleName));

        builder.setInsertionPointToStart(&moduleOp.body().front());

        // save module theModule
        auto parentModule = theModule;
        theModule = moduleOp;

        GenContext moduleGenContext = {0};
        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        // restore
        theModule = parentModule;

        builder.setInsertionPointAfter(moduleOp);

        return result;
    }

    mlir::LogicalResult mlirGenBody(Node body, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)body;
        if (kind == SyntaxKind::Block)
        {
            return mlirGen(body.as<Block>(), genContext);
        }

        if (kind == SyntaxKind::ModuleBlock)
        {
            return mlirGen(body.as<ModuleBlock>(), genContext);
        }

        if (body.is<Statement>())
        {
            return mlirGen(body.as<Statement>(), genContext);
        }

        if (body.is<Expression>())
        {
            auto result = mlirGen(body.as<Expression>(), genContext);
            if (result)
            {
                return mlirGenReturnValue(loc(body), result, genContext);
            }

            builder.create<mlir_ts::ReturnOp>(loc(body));
            return mlir::success();
        }

        llvm_unreachable("unknown body type");
    }

    mlir::LogicalResult mlirGen(ModuleBlock moduleBlockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        // clear up state
        for (auto &statement : moduleBlockAST->statements)
        {
            statement->processed = false;
        }

        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : moduleBlockAST->statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContext)))
                {
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
                theModule.emitError("can't resolve dependencies in namespace");
                return mlir::failure();
            }
        } while (notResolved > 0);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(Block blockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

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
        }

        for (auto &statement : blockAST->statements)
        {
            if (failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
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
        else if (kind == SyntaxKind::LabeledStatement)
        {
            return mlirGen(statementAST.as<LabeledStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAliasDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<TypeAliasDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::Block)
        {
            return mlirGen(statementAST.as<Block>(), genContext);
        }
        else if (kind == SyntaxKind::EnumDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<EnumDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ClassDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ClassLikeDeclaration>(), genContext);
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
        else if (kind == SyntaxKind::ModuleDeclaration)
        {
            return mlirGen(statementAST.as<ModuleDeclaration>(), genContext);
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
        mlirGen(expressionStatementAST->expression, genContext);
        return mlir::success();
    }

    mlir::Value mlirGen(Expression expressionAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)expressionAST;
        if (kind == SyntaxKind::NumericLiteral)
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
        else if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(expressionAST.as<Identifier>(), genContext);
        }
        else if (kind == SyntaxKind::CallExpression)
        {
            return mlirGen(expressionAST.as<CallExpression>(), genContext);
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
        else if (kind == SyntaxKind::PropertyAccessExpression)
        {
            return mlirGen(expressionAST.as<PropertyAccessExpression>(), genContext);
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
            return mlir::Value();
        }
        else if (kind == SyntaxKind::ThisKeyword)
        {
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
        else if (kind == SyntaxKind::Unknown /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::Value();
        }
        else if (kind == SyntaxKind::OmittedExpression /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::Value();
        }

        llvm_unreachable("unknown expression");
    }

    bool registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                          std::function<std::pair<mlir::Type, mlir::Value>()> func, const GenContext &genContext)
    {
        auto isGlobalScope = !genContext.funcOp; /*symbolTable.getCurScope()->getParentScope() == nullptr*/
        auto isGlobal = isGlobalScope || varClass == VariableClass::Var;
        auto isConst = varClass == VariableClass::Const || varClass == VariableClass::ConstRef;

        auto effectiveName = name;

        mlir::Value variableOp;
        mlir::Type varType;
        if (!isGlobal)
        {
            auto res = func();
            auto type = std::get<0>(res);
            auto init = std::get<1>(res);
            if (!type && genContext.allowPartialResolve)
            {
                return false;
            }

            assert(type);
            varType = type;

            if (isConst)
            {
                variableOp = init;
                // special cast to support ForOf
                if (varClass == VariableClass::ConstRef)
                {
                    MLIRCodeLogic mcl(builder);
                    variableOp = mcl.GetReferenceOfLoadOp(init);
                    if (!variableOp)
                    {
                        // convert ConstRef to Const again as this is const object (it seems)
                        variableOp = init;
                        varClass = VariableClass::Const;
                    }
                }
            }
            else
            {
                assert(type);

                MLIRTypeHelper mth(builder.getContext());

                auto copyRequired = false;
                auto actualType = mth.convertConstTypeToType(type, copyRequired);
                if (init && actualType != type)
                {
                    auto castValue = cast(location, actualType, init, genContext);
                    init = castValue;
                }

                varType = actualType;

                variableOp =
                    builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(actualType), init, builder.getBoolAttr(false));
            }
        }
        else
        {
            mlir_ts::GlobalOp globalOp;
            // get constant
            {
                mlir::OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(theModule.getBody());
                // find last string
                auto lastUse = [&](mlir::Operation *op) {
                    if (auto globalOp = dyn_cast_or_null<mlir_ts::GlobalOp>(op))
                    {
                        builder.setInsertionPointAfter(globalOp);
                    }
                };

                theModule.getBody()->walk(lastUse);

                effectiveName = getFullNamespaceName(name);

                globalOp = builder.create<mlir_ts::GlobalOp>(location,
                                                             // temp type
                                                             builder.getI32Type(), isConst, effectiveName, mlir::Attribute());

                if (isGlobalScope)
                {
                    auto &region = globalOp.getInitializerRegion();
                    auto *block = builder.createBlock(&region);

                    builder.setInsertionPoint(block, block->begin());

                    auto res = func();
                    auto type = std::get<0>(res);
                    auto init = std::get<1>(res);
                    if (!type && genContext.allowPartialResolve)
                    {
                        return false;
                    }

                    assert(type);
                    varType = type;

                    globalOp.typeAttr(mlir::TypeAttr::get(type));

                    // add return
                    if (init)
                    {
                        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{init});
                    }
                    else
                    {
                        auto undef = builder.create<mlir_ts::UndefOp>(location, type);
                        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{undef});
                    }
                }
            }

            if (!isGlobalScope)
            {
                auto res = func();
                auto type = std::get<0>(res);
                auto init = std::get<1>(res);
                if (!type && genContext.allowPartialResolve)
                {
                    return false;
                }

                assert(type);
                varType = type;

                globalOp.typeAttr(mlir::TypeAttr::get(type));

                // save value
                auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(type), name);
                builder.create<mlir_ts::StoreOp>(location, init, address);
            }
        }

        if (variableOp)
        {
            LLVM_DEBUG(dbgs() << "\n +++== variable = " << effectiveName << " type: " << varType << " op: " << variableOp << "==+++ \n";);
        }

        auto varDecl = std::make_shared<VariableDeclarationDOM>(effectiveName, varType, location);
        if (!isConst || varClass == VariableClass::ConstRef)
        {
            varDecl->setReadWriteAccess();
        }

        varDecl->setFuncOp(genContext.funcOp);

        if (!isGlobal)
        {
            declare(varDecl, variableOp);
        }
        else if (isFullName)
        {
            fullNameGlobalsMap.insert(name, varDecl);
        }
        else
        {
            getGlobalsMap().insert({name, varDecl});
        }

        return true;
    }

    template <typename ItemTy>
    bool processDeclaration(ItemTy item, VariableClass varClass, std::function<std::pair<mlir::Type, mlir::Value>()> func,
                            const GenContext &genContext)
    {
        auto location = loc(item);

        if (item->name == SyntaxKind::ArrayBindingPattern)
        {
            auto res = func();
            auto type = std::get<0>(res);
            auto init = std::get<1>(res);

            auto arrayBindingPattern = item->name.template as<ArrayBindingPattern>();
            auto index = 0;
            for (auto arrayBindingElement : arrayBindingPattern->elements)
            {
                MLIRPropertyAccessCodeLogic cl(builder, location, init, builder.getI32IntegerAttr(index++));
                mlir::Value subInit;
                TypeSwitch<mlir::Type>(type)
                    .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { subInit = cl.Tuple(constTupleType, true); })
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { subInit = cl.Tuple(tupleType, true); })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                if (!processDeclaration(
                        arrayBindingElement.template as<BindingElement>(), varClass,
                        [&]() { return std::make_pair(subInit.getType(), subInit); }, genContext))
                {
                    return false;
                }
            }
        }
        else
        {
            // name
            auto name = MLIRHelper::getName(item->name);

            // register
            return registerVariable(location, name, false, varClass, func, genContext);
        }

        return true;
    }

    template <typename ItemTy> std::pair<mlir::Type, mlir::Value> getTypeAndInit(ItemTy item, const GenContext &genContext)
    {
        // type
        mlir::Type type;
        if (item->type)
        {
            type = getType(item->type);
        }

        // init
        mlir::Value init;
        if (auto initializer = item->initializer)
        {
            init = mlirGen(initializer, genContext);
            if (init)
            {
                if (!type)
                {
                    type = init.getType();
                }
                else if (type != init.getType())
                {
                    auto castValue = cast(loc(initializer), type, init, genContext);
                    init = castValue;
                }
            }
        }

        return std::make_pair(type, init);
    }

    mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
    {
        auto isLet = (variableDeclarationListAST->flags & NodeFlags::Let) == NodeFlags::Let;
        auto isConst = (variableDeclarationListAST->flags & NodeFlags::Const) == NodeFlags::Const;
        auto varClass = isLet ? VariableClass::Let : isConst ? VariableClass::Const : VariableClass::Var;

        for (auto &item : variableDeclarationListAST->declarations)
        {
            if (!item->type && !item->initializer)
            {
                auto name = MLIRHelper::getName(item->name);
                emitError(loc(item)) << "type of variable '" << name << "' is not provided, variable must have type or initializer";
                return mlir::failure();
            }

            auto initFunc = [&]() { return getTypeAndInit(item, genContext); };

            auto valClassItem = varClass;
            if ((item->transformFlags & TransformFlags::ForceConstRef) == TransformFlags::ForceConstRef)
            {
                valClassItem = VariableClass::ConstRef;
            }

            if (!processDeclaration(item, valClassItem, initFunc, genContext))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
    {
        return mlirGen(variableStatementAST->declarationList, genContext);
    }

    std::vector<std::shared_ptr<FunctionParamDOM>> mlirGenParameters(SignatureDeclarationBase parametersContextAST,
                                                                     const GenContext &genContext)
    {
        std::vector<std::shared_ptr<FunctionParamDOM>> params;
        if (!parametersContextAST)
        {
            return params;
        }

        // add this param
        auto isStatic = hasModifier(parametersContextAST, SyntaxKind::StaticKeyword);
        if (!isStatic && (parametersContextAST == SyntaxKind::MethodDeclaration || parametersContextAST == SyntaxKind::Constructor ||
                          parametersContextAST == SyntaxKind::GetAccessor || parametersContextAST == SyntaxKind::SetAccessor))
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }

        if (parametersContextAST->parent.is<InterfaceDeclaration>())
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, getAnyType(), loc(parametersContextAST)));
        }

        auto noneType = mlir::NoneType::get(builder.getContext());

        auto formalParams = parametersContextAST->parameters;
        auto index = 0;
        for (auto arg : formalParams)
        {
            auto name = MLIRHelper::getName(arg->name);
            mlir::Type type;
            auto isOptional = !!arg->questionToken;
            auto typeParameter = arg->type;
            if (typeParameter)
            {
                type = getType(typeParameter);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                // we need to add temporary block
                auto tempFuncType = builder.getFunctionType(llvm::None, llvm::None);
                auto tempFuncOp = mlir::FuncOp::create(loc(initializer), name, tempFuncType);
                auto &entryBlock = *tempFuncOp.addEntryBlock();

                auto insertPoint = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(&entryBlock);

                auto initValue = mlirGen(initializer, genContext);
                if (initValue)
                {
                    // TODO: set type if not provided
                    isOptional = true;
                    if (!type || type == noneType)
                    {
                        auto baseType = initValue.getType();
                        // type = OptionalType::get(baseType);
                        type = baseType;
                    }
                }

                // remove temp block
                builder.restoreInsertionPoint(insertPoint);
                entryBlock.erase();
            }

            if ((!type || type == noneType) && genContext.argTypeDestFuncType)
            {
                type = genContext.argTypeDestFuncType.cast<mlir::FunctionType>().getInput(index);

                LLVM_DEBUG(dbgs() << "\n ...param " << name << " mapped to type " << type << "\n\n");
            }

            if (!type || type == noneType)
            {
                if (!genContext.allowPartialResolve)
                {
                    if (!typeParameter && !initializer)
                    {
                        auto funcName = MLIRHelper::getName(parametersContextAST->name);
                        emitError(loc(arg)) << "type of parameter '" << name
                                            << "' is not provided, parameter must have type or initializer, function: " << funcName;
                        return params;
                    }

                    emitError(loc(typeParameter)) << "can't resolve type for parameter '" << name << "'";
                }

                return params;
            }

            params.push_back(std::make_shared<FunctionParamDOM>(name, type, loc(arg), isOptional, initializer));

            index++;
        }

        return params;
    }

    std::tuple<std::string, std::string> getNameOfFunction(SignatureDeclarationBase signatureDeclarationBaseAST,
                                                           const GenContext &genContext)
    {
        std::string fullName = MLIRHelper::getName(signatureDeclarationBaseAST->name);
        std::string objectOwnerName;
        if (signatureDeclarationBaseAST->parent.is<ClassDeclaration>())
        {
            objectOwnerName = MLIRHelper::getName(signatureDeclarationBaseAST->parent.as<ClassDeclaration>()->name);
        }
        else if (signatureDeclarationBaseAST->parent.is<InterfaceDeclaration>())
        {
            objectOwnerName = MLIRHelper::getName(signatureDeclarationBaseAST->parent.as<InterfaceDeclaration>()->name);
        }

        if (signatureDeclarationBaseAST == SyntaxKind::MethodDeclaration || signatureDeclarationBaseAST == SyntaxKind::MethodSignature)
        {
            // class method name
            fullName = objectOwnerName + "." + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::GetAccessor)
        {
            // class method name
            fullName = objectOwnerName + ".get_" + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::SetAccessor)
        {
            // class method name
            fullName = objectOwnerName + ".set_" + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::Constructor)
        {
            // class method name
            fullName = objectOwnerName + "." + CONSTRUCTOR_NAME;
        }

        auto name = fullName;
        if (fullName.empty())
        {
            // auto calculate name
            std::stringstream ssName;
            ssName << "__uf" << hash_value(loc(signatureDeclarationBaseAST));
            name = fullName = ssName.str();
        }
        else
        {
            fullName = getFullNamespaceName(name).str();
        }

        return std::make_tuple(fullName, name);
    }

    std::tuple<FunctionPrototypeDOM::TypePtr, mlir::FunctionType, SmallVector<mlir::Type>> mlirGenFunctionSignaturePrototype(
        SignatureDeclarationBase signatureDeclarationBaseAST, const GenContext &genContext)
    {
        auto res = getNameOfFunction(signatureDeclarationBaseAST, genContext);
        auto fullName = std::get<0>(res);
        auto name = std::get<1>(res);

        auto params = mlirGenParameters(signatureDeclarationBaseAST, genContext);
        SmallVector<mlir::Type> argTypes;
        auto argNumber = 0;

        mlir::FunctionType funcType;

        for (const auto &param : params)
        {
            auto paramType = param->getType();
            if (!paramType)
            {
                return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
            }

            if (param->getIsOptional())
            {
                argTypes.push_back(getOptionalType(paramType));
            }
            else
            {
                argTypes.push_back(paramType);
            }

            argNumber++;
        }

        auto funcProto = std::make_shared<FunctionPrototypeDOM>(fullName, params);

        funcProto->setNameWithoutNamespace(name);

        // check if function already discovered
        auto funcIt = getFunctionMap().find(name);
        if (funcIt != getFunctionMap().end())
        {
            auto cachedFuncType = funcIt->second.getType();
            if (cachedFuncType.getNumResults() > 0)
            {
                auto returnType = cachedFuncType.getResult(0);
                funcProto->setReturnType(returnType);
            }

            funcType = cachedFuncType;
        }
        else if (auto typeParameter = signatureDeclarationBaseAST->type)
        {
            auto returnType = getType(typeParameter);
            funcProto->setReturnType(returnType);

            funcType = builder.getFunctionType(argTypes, returnType);
        }

        return std::make_tuple(funcProto, funcType, argTypes);
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, bool> mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        mlir_ts::FuncOp funcOp;

        auto res = mlirGenFunctionSignaturePrototype(functionLikeDeclarationBaseAST, genContext);
        auto funcProto = std::get<0>(res);
        if (!funcProto)
        {
            return std::make_tuple(funcOp, funcProto, false);
        }

        auto funcType = std::get<1>(res);
        auto argTypes = std::get<2>(res);
        auto fullName = funcProto->getName();

        // discover type & args
        if (!funcType)
        {
            if (mlir::succeeded(
                    discoverFunctionReturnTypeAndCapturedVars(functionLikeDeclarationBaseAST, fullName, argTypes, funcProto, genContext)))
            {
                // rewrite ret type with actual value
                if (auto typeParameter = functionLikeDeclarationBaseAST->type)
                {
                    auto returnType = getType(typeParameter);
                    funcProto->setReturnType(returnType);
                }
                else if (genContext.argTypeDestFuncType && genContext.argTypeDestFuncType.cast<mlir::FunctionType>().getNumResults() > 0)
                {
                    funcProto->setReturnType(genContext.argTypeDestFuncType.cast<mlir::FunctionType>().getResult(0));
                }

                // create funcType
                if (funcProto->getReturnType())
                {
                    funcType = builder.getFunctionType(argTypes, funcProto->getReturnType());
                }
                else
                {
                    // no return type
                    funcType = builder.getFunctionType(argTypes, llvm::None);
                }
            }
            else
            {
                // false result
                return std::make_tuple(funcOp, funcProto, false);
            }
        }

        auto it = getCaptureVarsMap().find(funcProto->getName());
        auto hasCapturedVars = funcProto->getHasCapturedVars() || (it != getCaptureVarsMap().end());
        if (hasCapturedVars)
        {
            SmallVector<mlir::NamedAttribute> attrs;
            SmallVector<mlir::DictionaryAttr> argAttrs;

            for (auto argType : funcType.getInputs())
            {
                SmallVector<mlir::NamedAttribute> argAttrsForType;
                // add nested to first attr
                if (argAttrs.size() == 0)
                {
                    // we need to force LLVM converter to allow to amend op in attached interface
                    attrs.push_back({builder.getIdentifier(TS_NEST_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
                    argAttrsForType.push_back({builder.getIdentifier(TS_NEST_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
                }

                auto argDicAttr = mlir::DictionaryAttr::get(builder.getContext(), argAttrsForType);
                argAttrs.push_back(argDicAttr);
            }

            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs, argAttrs);
        }
        else
        {
            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType);
        }

        return std::make_tuple(funcOp, funcProto, true);
    }

    mlir::LogicalResult discoverFunctionReturnTypeAndCapturedVars(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                                  StringRef name, SmallVector<mlir::Type> &argTypes,
                                                                  const FunctionPrototypeDOM::TypePtr &funcProto,
                                                                  const GenContext &genContext)
    {
        if (funcProto->getDiscovered())
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "??? discovering 'ret type' & 'captured vars' for : " << name << "\n";);

        mlir::OpBuilder::InsertionGuard guard(builder);

        auto partialDeclFuncType = builder.getFunctionType(argTypes, llvm::None);
        auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

        {
            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            GenContext genContextWithPassResult = {0};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.thisType = genContext.thisType;
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = new SmallVector<mlir::Block *>();
            genContextWithPassResult.passResult = new PassResult();
            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, dummyFuncOp, funcProto, genContextWithPassResult)))
            {
                auto &passResult = genContextWithPassResult.passResult;
                if (!passResult->functionReturnType && passResult->functionReturnTypeShouldBeProvided)
                {
                    // has return value but type is not provided yet
                    genContextWithPassResult.clean();
                    return mlir::failure();
                }

                funcProto->setDiscovered(true);
                auto discoveredType = passResult->functionReturnType;
                if (discoveredType && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?
                    MLIRTypeHelper mth(builder.getContext());
                    bool copyRequired;
                    funcProto->setReturnType(mth.convertConstTypeToType(discoveredType, copyRequired));
                    LLVM_DEBUG(llvm::dbgs() << "ret type: " << funcProto->getReturnType() << ", name: " << name << "\n";);
                }

                // if we have captured parameters, add first param to send lambda's type(class)
                if (passResult->outerVariables.size() > 0)
                {
                    MLIRCodeLogic mcl(builder);
                    argTypes.insert(argTypes.begin(), mcl.CaptureType(passResult->outerVariables));
                    getCaptureVarsMap().insert({name, passResult->outerVariables});

                    funcProto->setHasCapturedVars(true);

                    LLVM_DEBUG(llvm::dbgs() << "has captured vars, name: " << name << "\n";);
                }

                genContextWithPassResult.clean();
                return mlir::success();
            }
            else
            {
                genContextWithPassResult.clean();
                return mlir::failure();
            }
        }
    }

    mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        if (mlirGenFunctionLikeDeclaration(functionDeclarationAST, genContext))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::Value mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
    {
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            funcOp = mlirGenFunctionLikeDeclaration(functionExpressionAST, genContext);
            if (!funcOp)
            {
                return mlir::Value();
            }
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(loc(functionExpressionAST), funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    mlir::Value mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
    {
        auto location = loc(arrowFunctionAST);
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            funcOp = mlirGenFunctionLikeDeclaration(arrowFunctionAST, genContext);
            if (!funcOp)
            {
                return mlir::Value();
            }
        }

        if (auto trampOp = resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getType(), genContext))
        {
            return trampOp;
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(location, funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    mlir_ts::FuncOp mlirGenFunctionLikeDeclaration(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(functionLikeDeclarationBaseAST);

        auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, genContext);

        auto &funcOp = std::get<0>(funcOpWithFuncProto);
        auto &funcProto = std::get<1>(funcOpWithFuncProto);
        auto result = std::get<2>(funcOpWithFuncProto);
        if (!result || !funcOp)
        {
            return funcOp;
        }

        auto funcGenContext = GenContext(genContext);
        funcGenContext.funcOp = funcOp;
        funcGenContext.passResult = nullptr;

        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it != getCaptureVarsMap().end())
        {
            funcGenContext.capturedVars = &it->getValue();
        }

        auto resultFromBody = mlirGenFunctionBody(functionLikeDeclarationBaseAST, funcOp, funcProto, funcGenContext);
        if (mlir::failed(resultFromBody))
        {
            return funcOp;
        }

        // set visibility index
        if (funcProto->getName() != MAIN_ENTRY_NAME &&
            !hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ExportKeyword) /* && !funcProto->getNoBody()*/)
        {
            funcOp.setPrivate();
        }

        if (!genContext.dummyRun)
        {
            theModule.push_back(funcOp);
        }

        auto name = funcProto->getNameWithoutNamespace();
        if (!getFunctionMap().count(name))
        {
            getFunctionMap().insert({name, funcOp});

            LLVM_DEBUG(llvm::dbgs() << "\n... reg. func: " << name << " type:" << funcOp.getType() << "\n";);
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n... re-process. func: " << name << " type:" << funcOp.getType() << "\n";);
        }

        builder.setInsertionPointAfter(funcOp);

        return funcOp;
    }

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        auto retType = funcProto->getReturnType();
        auto hasReturn = retType && !retType.isa<mlir_ts::VoidType>();
        if (hasReturn)
        {
            auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
            auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
            varDecl->setReadWriteAccess();
            declare(varDecl, entryOp.reference());
        }
        else
        {
            builder.create<mlir_ts::EntryOp>(location, mlir::Type());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionExit(mlir::Location location, const GenContext &genContext)
    {
        builder.create<mlir_ts::ExitOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionThisParam(mlir::Location loc, int &firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                 mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        firstIndex++;
        auto capturedVars = *genContext.capturedVars;

        auto thisParam = arguments[firstIndex];
        auto thisRefType = thisParam.getType();

        auto thisParamVar = std::make_shared<VariableDeclarationDOM>(THIS_NAME, thisRefType, loc);

        declare(thisParamVar, thisParam);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionParams(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        auto index = firstIndex;
        for (const auto &param : funcProto->getArgs())
        {
            index++;
            mlir::Value paramValue;

            // alloc all args
            // process optional parameters
            if (param->getIsOptional() || param->hasInitValue())
            {
                // process init expression
                auto location = param->getLoc();

                auto optType = getOptionalType(param->getType());

                auto paramOptionalOp = builder.create<mlir_ts::ParamOptionalOp>(location, mlir_ts::RefType::get(optType), arguments[index]);

                paramValue = paramOptionalOp;

                if (param->hasInitValue())
                {
                    /*auto *defValueBlock =*/builder.createBlock(&paramOptionalOp.defaultValueRegion());

                    mlir::Value defaultValue;
                    auto initExpression = param->getInitValue();
                    if (initExpression)
                    {
                        defaultValue = mlirGen(initExpression, genContext);
                    }
                    else
                    {
                        llvm_unreachable("unknown statement");
                    }

                    if (optType != defaultValue.getType())
                    {
                        defaultValue = cast(location, optType, defaultValue, genContext);
                    }

                    builder.create<mlir_ts::ParamDefaultValueOp>(location, defaultValue);

                    builder.setInsertionPointAfter(paramOptionalOp);
                }
            }
            else
            {
                paramValue = builder.create<mlir_ts::ParamOp>(param->getLoc(), mlir_ts::RefType::get(param->getType()), arguments[index]);
            }

            if (paramValue)
            {
                // redefine variable
                param->setReadWriteAccess();
                declare(param, paramValue, true);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCaptures(FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
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
            auto _this = nf.createIdentifier(stows(THIS_NAME));
            auto _name = nf.createIdentifier(stows(std::string(name)));
            auto _this_name = nf.createPropertyAccessExpression(_this, _name);
            auto thisVarValue = mlirGen(_this_name, genContext);
            auto variableRefType = mlir_ts::RefType::get(variableInfo->getType());

            auto capturedParam = std::make_shared<VariableDeclarationDOM>(name, variableRefType, variableInfo->getLoc());
            if (thisVarValue.getType().isa<mlir_ts::RefType>())
            {
                capturedParam->setReadWriteAccess();
            }

            LLVM_DEBUG(dbgs() << "\n --- captured 'this->" << name << "' [" << thisVarValue << "] ref val type: " << variableRefType
                              << "\n\n");

            declare(capturedParam, thisVarValue);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir_ts::FuncOp funcOp,
                                            FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        if (!functionLikeDeclarationBaseAST->body)
        {
            // it is just declaration
            funcProto->setNoBody(true);
            return mlir::success();
        }

        auto location = loc(functionLikeDeclarationBaseAST);

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        // process function params
        for (auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
        {
            if (failed(declare(std::get<0>(paramPairs), std::get<1>(paramPairs))))
            {
                return mlir::failure();
            }
        }

        // allocate all params

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();
        auto firstIndex = -1;

        // add exit code
        if (failed(mlirGenFunctionEntry(location, funcProto, genContext)))
        {
            return mlir::failure();
        }

        // register this if lambda function
        if (failed(mlirGenFunctionThisParam(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCaptures(funcProto, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, genContext)))
        {
            return mlir::failure();
        }

        // add exit code
        if (failed(mlirGenFunctionExit(location, genContext)))
        {
            return mlir::failure();
        }

        if (genContext.dummyRun)
        {
            genContext.cleanUps->push_back(blockPtr);
        }

        return mlir::success();
    }

    mlir::Value mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext)
    {
        auto location = loc(typeAssertionAST);

        auto typeInfo = getType(typeAssertionAST->type);
        auto exprValue = mlirGen(typeAssertionAST->expression, genContext);

        auto castedValue = cast(location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    mlir::Value mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
    {
        auto location = loc(asExpressionAST);

        auto typeInfo = getType(asExpressionAST->type);
        auto exprValue = mlirGen(asExpressionAST->expression, genContext);

        auto castedValue = cast(location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
    {
        auto location = loc(returnStatementAST);
        if (auto expression = returnStatementAST->expression)
        {
            auto expressionValue = mlirGen(expression, genContext);
            return mlirGenReturnValue(location, expressionValue, genContext);
        }

        builder.create<mlir_ts::ReturnOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGenReturnValue(mlir::Location location, mlir::Value expressionValue, const GenContext &genContext)
    {
        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        // empty return
        if (!expressionValue)
        {
            if (genContext.allowPartialResolve)
            {
                // do not remove it, needed to process recursive functions
                return mlir::success();
            }

            return mlir::failure();
        }

        auto funcOp = const_cast<GenContext &>(genContext).funcOp;
        if (funcOp)
        {
            auto countResults = funcOp.getCallableResults().size();
            if (countResults > 0)
            {
                auto returnType = funcOp.getCallableResults().front();
                if (returnType != expressionValue.getType())
                {
                    auto castValue = cast(location, returnType, expressionValue, genContext);
                    expressionValue = castValue;
                }
            }
        }

        // record return type if not provided
        if (genContext.passResult)
        {
            genContext.passResult->functionReturnType = expressionValue.getType();
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

        builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(ifStatementAST);

        auto hasElse = !!ifStatementAST->elseStatement;

        // condition
        auto condValue = mlirGen(ifStatementAST->expression, genContext);
        if (genContext.allowPartialResolve && !condValue)
        {
            return mlir::failure();
        }

        if (condValue.getType() != getBooleanType())
        {
            condValue = cast(location, getBooleanType(), condValue, genContext);
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        mlirGen(ifStatementAST->thenStatement, genContext);

        if (hasElse)
        {
            builder.setInsertionPointToStart(&ifOp.elseRegion().front());
            mlirGen(ifStatementAST->elseStatement, genContext);
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

        /*auto *cond =*/builder.createBlock(&doWhileOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&doWhileOp.body(), {}, types);

        // body in condition
        builder.setInsertionPointToStart(&doWhileOp.body().front());
        mlirGen(doStatementAST->statement, genContext);
        // just simple return, as body in cond
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointToStart(&doWhileOp.cond().front());
        auto conditionValue = mlirGen(doStatementAST->expression, genContext);
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

        /*auto *cond =*/builder.createBlock(&whileOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&whileOp.body(), {}, types);

        // condition
        builder.setInsertionPointToStart(&whileOp.cond().front());
        auto conditionValue = mlirGen(whileStatementAST->expression, genContext);
        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        // body
        builder.setInsertionPointToStart(&whileOp.body().front());
        mlirGen(whileStatementAST->statement, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(whileOp);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForStatement forStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forStatementAST);

        // initializer
        // TODO: why do we have ForInitialier
        if (forStatementAST->initializer.is<Expression>())
        {
            auto init = mlirGen(forStatementAST->initializer.as<Expression>(), genContext);
            if (!init)
            {
                return mlir::failure();
            }
        }
        else if (forStatementAST->initializer.is<VariableDeclarationList>())
        {
            auto result = mlirGen(forStatementAST->initializer.as<VariableDeclarationList>(), genContext);
            if (failed(result))
            {
                return result;
            }
        }

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto forOp = builder.create<mlir_ts::ForOp>(location, types, operands);
        if (!label.empty())
        {
            forOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        /*auto *cond =*/builder.createBlock(&forOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&forOp.body(), {}, types);
        /*auto *incr =*/builder.createBlock(&forOp.incr(), {}, types);

        builder.setInsertionPointToStart(&forOp.cond().front());
        auto conditionValue = mlirGen(forStatementAST->condition, genContext);
        if (conditionValue)
        {
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});
        }
        else
        {
            builder.create<mlir_ts::NoConditionOp>(location, mlir::ValueRange{});
        }

        // body
        builder.setInsertionPointToStart(&forOp.body().front());
        mlirGen(forStatementAST->statement, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        // increment
        builder.setInsertionPointToStart(&forOp.incr().front());
        mlirGen(forStatementAST->incrementor, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(forOp);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForInStatement forInStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forInStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S("_i_"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S("_a_"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forInStatementAST->expression);
        arrayVar->transformFlags |= TransformFlags::ForceConstRef;
        declarations.push_back(arrayVar);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        // condition
        // auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
        // nf.createCallExpression(nf.createIdentifier(S("#_last_field")), undefined, NodeArray<Expression>(_a)));
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S("length"))));

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

    mlir::LogicalResult mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S("_i_"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S("_a_"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forOfStatementAST->expression);
        arrayVar->transformFlags |= TransformFlags::ForceConstRef;
        declarations.push_back(arrayVar);

        // condition
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S("length"))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = nf.createElementAccessExpression(_a, _i);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let /*varDeclList->flags*/);

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext)
    {
        auto location = loc(labeledStatementAST);

        label = MLIRHelper::getName(labeledStatementAST->label);
        auto res = mlirGen(labeledStatementAST->statement, genContext);

        return res;
    }

    mlir::LogicalResult mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext)
    {
        auto location = loc(continueStatementAST);

        auto label = MLIRHelper::getName(continueStatementAST->label);

        builder.create<mlir_ts::ContinueOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(BreakStatement breakStatementAST, const GenContext &genContext)
    {
        auto location = loc(breakStatementAST);

        auto label = MLIRHelper::getName(breakStatementAST->label);

        builder.create<mlir_ts::BreakOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGenSwitchCase(mlir::Location location, mlir::Value switchValue, NodeArray<ts::CaseOrDefaultClause> &clauses,
                                          int index, mlir::Block *&lastBlock, mlir::Block *&lastConditionBlock, mlir::Block *mergeBlock,
                                          const GenContext &genContext)
    {
        auto caseBlock = clauses[index];
        auto statements = caseBlock->statements;
        // inline block
        if (statements.size() == 1)
        {
            auto firstStatement = statements.front();
            if ((SyntaxKind)firstStatement == SyntaxKind::Block)
            {
                statements = statements.front().as<Block>()->statements;
            }
        }

        mlir::Block *caseBodyBlock = nullptr;
        mlir::Block *caseConditionBlock = nullptr;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            caseBodyBlock = builder.createBlock(lastConditionBlock);

            auto hasBreak = false;
            for (auto statement : statements)
            {
                if ((SyntaxKind)statement == SyntaxKind::BreakStatement)
                {
                    hasBreak = true;
                    break;
                }

                mlirGen(statement, genContext);
            }

            // exit;
            builder.create<mlir::BranchOp>(location, hasBreak ? mergeBlock : lastBlock);

            lastBlock = caseBodyBlock;
        }

        switch ((SyntaxKind)caseBlock)
        {
        case SyntaxKind::CaseClause: {
            {

                mlir::OpBuilder::InsertionGuard guard(builder);
                caseConditionBlock = builder.createBlock(lastBlock);

                auto caseValue = mlirGen(caseBlock.as<CaseClause>()->expression, genContext);

                auto condition = builder.create<mlir_ts::LogicalBinaryOp>(
                    location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), switchValue, caseValue);

                auto conditionI1 = cast(location, builder.getI1Type(), condition, genContext);

                builder.create<mlir::CondBranchOp>(location, conditionI1, caseBodyBlock, /*trueArguments=*/mlir::ValueRange{},
                                                   lastConditionBlock, /*falseArguments=*/mlir::ValueRange{});

                lastConditionBlock = caseConditionBlock;
            }

            // create condition block
        }
        break;
        case SyntaxKind::DefaultClause:
            lastConditionBlock = lastBlock;
            break;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
    {
        auto location = loc(switchStatementAST);

        auto switchValue = mlirGen(switchStatementAST->expression, genContext);

        auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

        // add merge block
        switchOp.addMergeBlock();
        auto *mergeBlock = switchOp.getMergeBlock();

        auto *lastBlock = mergeBlock;
        auto *lastConditionBlock = mergeBlock;

        auto &clauses = switchStatementAST->caseBlock->clauses;

        // process default first (in our case it will be the last)
        for (int index = clauses.size() - 1; index >= 0; index--)
        {
            if (SyntaxKind::DefaultClause == (SyntaxKind)clauses[index])
                if (mlir::failed(
                        mlirGenSwitchCase(location, switchValue, clauses, index, lastBlock, lastConditionBlock, mergeBlock, genContext)))
                {
                    return mlir::failure();
                }
        }

        // process rest without default
        for (int index = clauses.size() - 1; index >= 0; index--)
        {
            if (SyntaxKind::DefaultClause != (SyntaxKind)clauses[index])
                if (mlir::failed(
                        mlirGenSwitchCase(location, switchValue, clauses, index, lastBlock, lastConditionBlock, mergeBlock, genContext)))
                {
                    return mlir::failure();
                }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
    {
        auto location = loc(throwStatementAST);

        auto exception = mlirGen(throwStatementAST->expression, genContext);

        builder.create<mlir_ts::ThrowOp>(location, exception);

        return mlir::success();

        // TODO: read about LLVM_ResumeOp,  maybe this is what you need (+LLVM_InvokeOp, LLVM_LandingpadOp)

        // TODO: PS, you can add param to each method to process return "exception info", and check every call for methods if they return
        // exception info

        /*
llvm.mlir.global external constant @_ZTIi() : !llvm.ptr<i8>
llvm.func @foo(!llvm.ptr<i8>)
llvm.func @bar(!llvm.ptr<i8>) -> !llvm.ptr<i8>
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = alloca i8
%0 = llvm.mlir.constant(0 : i32) : i32
%1 = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
%2 = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
%3 = llvm.bitcast %2 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
%4 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
%5 = llvm.mlir.constant(1 : i32) : i32
%6 = llvm.alloca %5 x i8 : (i32) -> !llvm.ptr<i8>
// CHECK: invoke void @foo(i8* %[[a1]])
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
llvm.invoke @foo(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> ()

// CHECK: [[unwind]]:
^bb1:
// CHECK: %{{[0-9]+}} = landingpad { i8*, i32 }
// CHECK-NEXT:             catch i8** null
// CHECK-NEXT:             catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:             filter [1 x i8] c"\01"
%7 = llvm.landingpad (catch %4 : !llvm.ptr<ptr<i8>>) (catch %3 : !llvm.ptr<i8>) (filter %1 : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>,
i32)>
// CHECK: br label %[[final:[0-9]+]]
llvm.br ^bb3

// CHECK: [[normal]]:
// CHECK-NEXT: ret i32 1
^bb2:	// 2 preds: ^bb0, ^bb3
llvm.return %5 : i32

// CHECK: [[final]]:
// CHECK-NEXT: %{{[0-9]+}} = invoke i8* @bar(i8* %[[a1]])
// CHECK-NEXT:          to label %[[normal]] unwind label %[[unwind]]
^bb3:	// pred: ^bb1
%8 = llvm.invoke @bar(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
}

        */
    }

    mlir::LogicalResult mlirGen(TryStatement tryStatementAST, const GenContext &genContext)
    {
        auto location = loc(tryStatementAST);

        const_cast<GenContext &>(genContext).funcOp.personalityAttr(builder.getBoolAttr(true));

        auto tryOp = builder.create<mlir_ts::TryOp>(location);

        SmallVector<mlir::Type, 0> types;

        /*auto *body =*/builder.createBlock(&tryOp.body(), {}, types);
        /*auto *catches =*/builder.createBlock(&tryOp.catches(), {}, types);
        /*auto *finallyBlock =*/builder.createBlock(&tryOp.finallyBlock(), {}, types);

        // body
        builder.setInsertionPointToStart(&tryOp.body().front());
        auto result = mlirGen(tryStatementAST->tryBlock, genContext);
        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // catches
        builder.setInsertionPointToStart(&tryOp.catches().front());
        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // finally
        builder.setInsertionPointToStart(&tryOp.finallyBlock().front());
        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(tryOp);
        return result;
    }

    mlir::Value mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
    {
        return mlirGen(unaryExpressionAST.as<Expression>(), genContext);
    }

    mlir::Value mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext)
    {
        return mlirGen(leftHandSideExpressionAST.as<Expression>(), genContext);
    }

    mlir::Value mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(prefixUnaryExpressionAST);

        auto opCode = prefixUnaryExpressionAST->_operator;

        auto expression = prefixUnaryExpressionAST->operand;
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE_EXPR(expressionValue, expression)

        auto boolValue = expressionValue;

        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:

            if (expressionValue.getType() != getBooleanType())
            {
                boolValue = cast(location, getBooleanType(), expressionValue, genContext);
            }

            return builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(), builder.getI32IntegerAttr((int)opCode),
                                                              boolValue);
        case SyntaxKind::TildeToken:
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
            return builder.create<mlir_ts::ArithmeticUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                              expressionValue);
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return builder.create<mlir_ts::PrefixUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                          expressionValue);
        default:
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(postfixUnaryExpressionAST);

        auto opCode = postfixUnaryExpressionAST->_operator;

        auto expression = postfixUnaryExpressionAST->operand;
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE_EXPR(expressionValue, expression)

        switch (opCode)
        {
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return builder.create<mlir_ts::PostfixUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                           expressionValue);
        default:
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext)
    {
        auto location = loc(conditionalExpressionAST);

        // condition
        auto condExpression = conditionalExpressionAST->condition;
        auto condValue = mlirGen(condExpression, genContext);

        VALIDATE_EXPR(condValue, condExpression);

        if (condValue.getType() != getBooleanType())
        {
            condValue = cast(location, getBooleanType(), condValue, genContext);
        }

        // detect value type
        mlir::Type resultType;
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto whenTrueExpression = conditionalExpressionAST->whenTrue;
            auto resultTrueTemp = mlirGen(whenTrueExpression, genContext);

            VALIDATE_EXPR(resultTrueTemp, whenTrueExpression);

            resultType = resultTrueTemp.getType();

            // it is temp calculation, remove it
            resultTrueTemp.getDefiningOp()->erase();
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto whenTrueExpression = conditionalExpressionAST->whenTrue;
        auto resultTrue = mlirGen(whenTrueExpression, genContext);

        VALIDATE_EXPR(resultTrue, whenTrueExpression);

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto whenFalseExpression = conditionalExpressionAST->whenFalse;
        auto resultFalse = mlirGen(whenFalseExpression, genContext);

        VALIDATE_EXPR(resultFalse, whenFalseExpression);

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    }

    mlir::Value mlirGenAndOrLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext, bool andOp)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        // condition
        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE_EXPR(leftExpressionValue, leftExpression)

        auto resultType = leftExpressionValue.getType();

        auto condValue = cast(location, getBooleanType(), leftExpressionValue, genContext);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto resultTrue = andOp ? mlirGen(rightExpression, genContext) : leftExpressionValue;

        if (andOp)
        {
            VALIDATE_EXPR(resultTrue, rightExpression)
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto resultFalse = andOp ? leftExpressionValue : mlirGen(rightExpression, genContext);

        if (!andOp)
        {
            VALIDATE_EXPR(resultFalse, rightExpression)
        }

        // sync right part
        if (resultTrue.getType() != resultFalse.getType())
        {
            resultFalse = cast(location, resultTrue.getType(), resultFalse, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    }

    mlir::Value mlirGenInLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        // Supports only array now
        auto location = loc(binaryExpressionAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // condition
        auto cond =
            nf.createBinaryExpression(binaryExpressionAST->left, nf.createToken(SyntaxKind::LessThanToken),
                                      nf.createPropertyAccessExpression(binaryExpressionAST->right, nf.createIdentifier(S("length"))));

        return mlirGen(cond, genContext);
    }

    mlir::Value mlirGenInstanceOfLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto result = mlirGen(binaryExpressionAST->left, genContext);
        auto resultType = result.getType();
        auto type = getTypeByTypeName(binaryExpressionAST->right, genContext);

        return builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(resultType == type));
    }

    mlir::Value evaluateBinaryOp(mlir::Location location, SyntaxKind opCode, mlir_ts::ConstantOp leftConstOp,
                                 mlir_ts::ConstantOp rightConstOp, const GenContext &genContext)
    {
        auto leftInt = leftConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
        auto rightInt = rightConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
        auto resultType = leftConstOp.getType();

        int64_t result = 0;
        switch (opCode)
        {
        case SyntaxKind::PlusEqualsToken:
            result = leftInt + rightInt;
            break;
        case SyntaxKind::LessThanLessThanToken:
            result = leftInt << rightInt;
            break;
        case SyntaxKind::GreaterThanGreaterThanToken:
            result = leftInt >> rightInt;
            break;
        case SyntaxKind::AmpersandToken:
            result = leftInt & rightInt;
            break;
        case SyntaxKind::BarToken:
            result = leftInt | rightInt;
            break;
        default:
            llvm_unreachable("not implemented");
            break;
        }

        leftConstOp.erase();
        rightConstOp.erase();

        return builder.create<mlir_ts::ConstantOp>(location, resultType, builder.getI64IntegerAttr(result));
    }

    mlir::Value mlirGenSaveLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE_EXPR(leftExpressionValue, leftExpression)

        if (auto funcType = leftExpressionValue.getType().dyn_cast_or_null<mlir::FunctionType>())
        {
            const_cast<GenContext &>(genContext).argTypeDestFuncType = funcType;
        }

        auto rightExpressionValue = mlirGen(rightExpression, genContext);

        VALIDATE_EXPR(rightExpressionValue, rightExpression)

        // clear state
        const_cast<GenContext &>(genContext).argTypeDestFuncType = nullptr;

        auto leftExpressionValueBeforeCast = leftExpressionValue;

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = cast(loc(rightExpression), getStringType(), rightExpressionValue, genContext);
            }
        }

        auto result = rightExpressionValue;

        // saving
        if (leftExpressionValueBeforeCast.getType() != result.getType())
        {
            result = cast(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result, genContext);
        }

        // TODO: finish it for field access, review CodeLogicHelper.saveResult
        if (auto loadOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::LoadOp>())
        {
            // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
            builder.create<mlir_ts::StoreOp>(location, result, loadOp.reference());
        }
        else if (auto accessorRefOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::AccessorRefOp>())
        {
            auto callRes = builder.create<mlir_ts::CallOp>(location, accessorRefOp.setAccessor().getValue(), mlir::TypeRange{getVoidType()},
                                                           mlir::ValueRange{result});
            result = callRes.getResult(0);
        }
        else if (auto thisAccessorRefOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisAccessorRefOp>())
        {
            auto callRes =
                builder.create<mlir_ts::CallOp>(location, thisAccessorRefOp.setAccessor().getValue(), mlir::TypeRange{getVoidType()},
                                                mlir::ValueRange{thisAccessorRefOp.thisVal(), result});
            result = callRes.getResult(0);
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return result;
    }

    mlir::Value mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto opCode = (SyntaxKind)binaryExpressionAST->operatorToken;

        auto saveResult = MLIRLogicHelper::isNeededToSaveData(opCode);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
        {
            return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken);
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

        auto leftExpressionValue = mlirGen(leftExpression, genContext);
        auto rightExpressionValue = mlirGen(rightExpression, genContext);

        VALIDATE_EXPR(rightExpressionValue, rightExpression)
        VALIDATE_EXPR(leftExpressionValue, leftExpression)

        // check if const expr.
        if (genContext.allowConstEval)
        {
            auto leftConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(leftExpressionValue.getDefiningOp());
            auto rightConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(rightExpressionValue.getDefiningOp());
            if (leftConstOp && rightConstOp)
            {
                // try to evaluate
                return evaluateBinaryOp(location, opCode, leftConstOp, rightConstOp, genContext);
            }
        }

        auto leftExpressionValueBeforeCast = leftExpressionValue;
        auto rightExpressionValueBeforeCast = rightExpressionValue;

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            // TODO: temporary hack
            if (leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                leftExpressionValue = cast(loc(leftExpression), getStringType(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = cast(loc(rightExpression), getStringType(), rightExpressionValue, genContext);
            }

            // end todo

            if (!MLIRLogicHelper::isLogicOp(opCode))
            {
                // cast from optional<T> type
                if (auto leftOptType = leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    leftExpressionValue = cast(loc(leftExpression), leftOptType.getElementType(), leftExpressionValue, genContext);
                }

                if (auto rightOptType = rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    rightExpressionValue = cast(loc(rightExpression), rightOptType.getElementType(), rightExpressionValue, genContext);
                }
            }
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
            // cast to int
            if (leftExpressionValue.getType() != builder.getI32Type())
            {
                leftExpressionValue = cast(loc(leftExpression), builder.getI32Type(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != builder.getI32Type())
            {
                rightExpressionValue = cast(loc(rightExpression), builder.getI32Type(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::SlashToken:
        case SyntaxKind::PercentToken:
        case SyntaxKind::AsteriskAsteriskToken:

            if (leftExpressionValue.getType() != builder.getF32Type())
            {
                leftExpressionValue = cast(loc(leftExpression), builder.getF32Type(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != builder.getF32Type())
            {
                rightExpressionValue = cast(loc(rightExpression), builder.getF32Type(), rightExpressionValue, genContext);
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

            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                // cast to base type
                auto hasF32 =
                    leftExpressionValue.getType() == builder.getF32Type() || rightExpressionValue.getType() == builder.getF32Type();
                if (hasF32)
                {
                    if (leftExpressionValue.getType() != builder.getF32Type())
                    {
                        leftExpressionValue = cast(loc(leftExpression), builder.getF32Type(), leftExpressionValue, genContext);
                    }

                    if (rightExpressionValue.getType() != builder.getF32Type())
                    {
                        rightExpressionValue = cast(loc(rightExpression), builder.getF32Type(), rightExpressionValue, genContext);
                    }
                }
                else
                {
                    auto hasI32 =
                        leftExpressionValue.getType() == builder.getI32Type() || rightExpressionValue.getType() == builder.getI32Type();
                    if (hasI32)
                    {
                        if (leftExpressionValue.getType() != builder.getI32Type())
                        {
                            leftExpressionValue = cast(loc(leftExpression), builder.getI32Type(), leftExpressionValue, genContext);
                        }

                        if (rightExpressionValue.getType() != builder.getI32Type())
                        {
                            rightExpressionValue = cast(loc(rightExpression), builder.getI32Type(), rightExpressionValue, genContext);
                        }
                    }
                }
            }

            break;
        default:
            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                rightExpressionValue = cast(loc(rightExpression), leftExpressionValue.getType(), rightExpressionValue, genContext);
            }

            break;
        }

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
            result = builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(), builder.getI32IntegerAttr((int)opCode),
                                                              leftExpressionValue, rightExpressionValue);
            break;
        case SyntaxKind::CommaToken:
            return rightExpressionValue;
        default:
            result = builder.create<mlir_ts::ArithmeticBinaryOp>(
                location, leftExpressionValue.getType(), builder.getI32IntegerAttr((int)opCode), leftExpressionValue, rightExpressionValue);
            break;
        }

        if (saveResult)
        {
            if (leftExpressionValueBeforeCast.getType() != result.getType())
            {
                result = cast(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result, genContext);
            }

            // TODO: finish it for field access, review CodeLogicHelper.saveResult
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValueBeforeCast.getDefiningOp()))
            {
                // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
                builder.create<mlir_ts::StoreOp>(location, result, loadOp.reference());
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return result;
    }

    mlir::Value mlirGen(SpreadElement spreadElement, const GenContext &genContext)
    {
        return mlirGen(spreadElement->expression, genContext);
    }

    mlir::Value mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
    {
        return mlirGen(parenthesizedExpression->expression, genContext);
    }

    mlir::Value mlirGen(QualifiedName qualifiedName, const GenContext &genContext)
    {
        auto location = loc(qualifiedName);

        auto expression = qualifiedName->left;
        auto expressionValue = mlirGenModuleReference(expression, genContext);

        VALIDATE_EXPR(expressionValue, expression)

        auto name = MLIRHelper::getName(qualifiedName->right);

        mlir::Value value;
        if (!expressionValue.getType() || expressionValue.getType() == mlir::NoneType::get(builder.getContext()))
        {
            if (auto namespaceRef = dyn_cast_or_null<mlir_ts::NamespaceRefOp>(expressionValue.getDefiningOp()))
            {
                // todo resolve namespace
                auto namespaceInfo = getNamespaceByFullName(namespaceRef.identifier());

                assert(namespaceInfo);

                auto saveNamespace = currentNamespace;
                currentNamespace = namespaceInfo;

                value = mlirGen(location, name, genContext);

                currentNamespace = saveNamespace;
            }

            return value;
        }

        emitError(location, "Can't resolve qualified name");

        llvm_unreachable("not implemented");
    }

    mlir::Value mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
    {
        auto location = loc(propertyAccessExpression);

        auto expression = propertyAccessExpression->expression.as<Expression>();
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE_EXPR(expressionValue, expression)

        auto name = MLIRHelper::getName(propertyAccessExpression->name);

        return mlirGenPropertyAccessExpression(location, expressionValue, name, genContext);
    }

    mlir::Value mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue, mlir::StringRef name,
                                                const GenContext &genContext)
    {
        mlir::Value value;

        assert(objectValue);

        // TODO: create NamespaceType instead of using this hacky code
        if (!objectValue.getType() || objectValue.getType() == mlir::NoneType::get(builder.getContext()))
        {
            if (auto namespaceRef = dyn_cast_or_null<mlir_ts::NamespaceRefOp>(objectValue.getDefiningOp()))
            {
                // todo resolve namespace
                auto namespaceInfo = getNamespaceByFullName(namespaceRef.identifier());

                assert(namespaceInfo);

                auto saveNamespace = currentNamespace;
                currentNamespace = namespaceInfo;

                value = mlirGen(location, name, genContext);

                currentNamespace = saveNamespace;
                return value;
            }

            LLVM_DEBUG(dbgs() << "mlirGenPropertyAccessExpression name: " << name << " object: " << objectValue << "\n");
            assert(genContext.allowPartialResolve);

            return value;
        }

        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, name);

        TypeSwitch<mlir::Type>(objectValue.getType())
            .Case<mlir_ts::EnumType>([&](auto enumType) { value = cl.Enum(enumType); })
            .Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { value = cl.Tuple(constTupleType); })
            .Case<mlir_ts::TupleType>([&](auto tupleType) { value = cl.Tuple(tupleType); })
            .Case<mlir_ts::BooleanType>([&](auto intType) { value = cl.Bool(intType); })
            .Case<mlir::IntegerType>([&](auto intType) { value = cl.Int(intType); })
            .Case<mlir::FloatType>([&](auto intType) { value = cl.Float(intType); })
            .Case<mlir_ts::StringType>([&](auto stringType) { value = cl.String(stringType); })
            .Case<mlir_ts::ConstArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::ArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::RefType>([&](auto refType) { value = cl.Ref(refType); })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                value = cl.TupleNoError(classStorageType);
                if (!value)
                {
                    value = ClassMembers(location, objectValue, classStorageType.getName().getValue(), name, true, true, genContext);
                }
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                value = cl.Class(classType);
                if (!value)
                {
                    value = ClassMembers(location, objectValue, classType.getName().getValue(), name, false, false, genContext);
                }
            })
            .Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                value = InterfaceMembers(location, objectValue, interfaceType.getName().getValue(), name, genContext);
            })
            .Default([](auto type) { llvm_unreachable("not implemented"); });

        if (value || genContext.allowPartialResolve)
        {
            return value;
        }

        emitError(location, "Can't resolve property name");

        llvm_unreachable("not implemented");
    }

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, mlir::StringRef classFullName, mlir::StringRef name,
                             bool baseClass, bool storageType, const GenContext &genContext)
    {
        auto classInfo = getClassByFullName(classFullName);
        assert(classInfo);

        // static field access
        auto value = ClassMembers(location, thisValue, classInfo, name, baseClass, storageType, genContext);
        if (!value && !genContext.allowPartialResolve)
        {
            emitError(location, "Class member '") << name << "' can't be found";
        }

        return value;
    }

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, ClassInfo::TypePtr classInfo, mlir::StringRef name,
                             bool baseClass, bool storageType, const GenContext &genContext)
    {
        assert(classInfo);

        MLIRCodeLogic mcl(builder);
        auto staticFieldIndex = classInfo->getStaticFieldIndex(mcl.TupleFieldName(name));
        if (staticFieldIndex >= 0)
        {
            auto fieldInfo = classInfo->staticFields[staticFieldIndex];
            auto value = resolveFullNameIdentifier(location, fieldInfo.globalVariableName, false, genContext);
            assert(value);
            return value;
        }

        // check method access
        auto methodIndex = classInfo->getMethodIndex(name);
        if (methodIndex >= 0)
        {
            auto methodInfo = classInfo->methods[methodIndex];
            auto funcOp = methodInfo.funcOp;
            auto effectiveFuncType = funcOp.getType();

            if (methodInfo.isStatic)
            {
                auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, effectiveFuncType,
                                                                   mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return symbOp;
            }
            else
            {
                auto effectiveThisValue = thisValue;
                if (baseClass)
                {
                    // get reference in case of classStorage
                    if (storageType)
                    {
                        MLIRCodeLogic mcl(builder);
                        thisValue = mcl.GetReferenceOfLoadOp(thisValue);
                        assert(thisValue);
                    }

                    effectiveThisValue = cast(location, classInfo->classType, thisValue, genContext);
                }

                if (methodInfo.isVirtual)
                {
                    // adding call of ctor
                    NodeFactory nf(NodeFactoryFlags::None);

                    auto vtableAccess = mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);

                    auto thisVirtualSymbOp = builder.create<mlir_ts::ThisVirtualSymbolRefOp>(
                        location, effectiveFuncType, effectiveThisValue, vtableAccess, builder.getI32IntegerAttr(methodInfo.virtualIndex),
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                    return thisVirtualSymbOp;
                }

                auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                    location, effectiveFuncType, effectiveThisValue, mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return thisSymbOp;
            }
        }

        // check accessor
        auto accessorIndex = classInfo->getAccessorIndex(name);
        if (accessorIndex >= 0)
        {
            auto accessorInfo = classInfo->accessors[accessorIndex];
            auto getFuncOp = accessorInfo.get;
            auto setFuncOp = accessorInfo.set;
            auto effectiveFuncType =
                getFuncOp ? getFuncOp.getType().dyn_cast_or_null<mlir::FunctionType>().getResult(0)
                          : setFuncOp.getType().dyn_cast_or_null<mlir::FunctionType>().getInput(accessorInfo.isStatic ? 0 : 1);

            if (accessorInfo.isStatic)
            {
                auto symbOp = builder.create<mlir_ts::AccessorRefOp>(
                    location, effectiveFuncType,
                    getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName()) : mlir::FlatSymbolRefAttr{},
                    setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName()) : mlir::FlatSymbolRefAttr{});
                return symbOp;
            }
            else
            {
                auto thisSymbOp = builder.create<mlir_ts::ThisAccessorRefOp>(
                    location, effectiveFuncType, thisValue,
                    getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName()) : mlir::FlatSymbolRefAttr{},
                    setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName()) : mlir::FlatSymbolRefAttr{});
                return thisSymbOp;
            }
        }

        auto first = true;
        for (auto baseClass : classInfo->baseClasses)
        {
            if (first && name == SUPER_NAME)
            {
                auto value = mlirGenPropertyAccessExpression(location, thisValue, baseClass->fullName, genContext);
                return value;
            }

            auto value = ClassMembers(location, thisValue, baseClass, name, true, false, genContext);
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
                    auto fieldValue = mlirGenPropertyAccessExpression(location, currentObject, chain->fullName, genContext);
                    if (!fieldValue)
                    {
                        if (!genContext.allowPartialResolve)
                        {
                            emitError(location) << "Can't resolve field/property/base '" << chain->fullName << "' of class '"
                                                << classInfo->fullName << "'\n";
                        }

                        return fieldValue;
                    }

                    assert(fieldValue);
                    currentObject = fieldValue;
                }

                // last value
                auto value = mlirGenPropertyAccessExpression(location, currentObject, name, genContext);
                if (value)
                {
                    return value;
                }
            }

            first = false;
        }

        if (baseClass || genContext.allowPartialResolve)
        {
            return mlir::Value();
        }

        emitError(location) << "can't resolve property/field/base '" << name << "' of class '" << classInfo->fullName << "'\n";

        assert(false);
        llvm_unreachable("not implemented");
    }

    bool classHasField(ClassInfo::TypePtr classInfo, mlir::StringRef name, SmallVector<ClassInfo::TypePtr> &fieldPath)
    {
        MLIRCodeLogic mcl(builder);

        auto fieldId = mcl.TupleFieldName(name);
        auto classStorageType = classInfo->classType.getStorageType().cast<mlir_ts::ClassStorageType>();
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
                                 mlir::StringRef name, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceByFullName(interfaceFullName);
        assert(interfaceInfo);

        // static field access
        auto value = InterfaceMembers(location, interfaceValue, interfaceInfo, name, genContext);
        if (!value && !genContext.allowPartialResolve)
        {
            emitError(location, "Interface member '") << name << "' can't be found";
        }

        return value;
    }

    mlir::Value InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, InterfaceInfo::TypePtr interfaceInfo,
                                 mlir::StringRef name, const GenContext &genContext)
    {
        assert(interfaceInfo);

        // check method access
        auto methodIndex = interfaceInfo->getMethodIndex(name);
        if (methodIndex >= 0)
        {
            auto methodInfo = interfaceInfo->methods[methodIndex];
            auto effectiveFuncType = methodInfo.funcType;

            // adding call of ctor
            NodeFactory nf(NodeFactoryFlags::None);

            auto interfaceSymbolRefOp = builder.create<mlir_ts::InterfaceSymbolRefOp>(
                location, effectiveFuncType, getAnyType(), interfaceValue, builder.getI32IntegerAttr(methodInfo.virtualIndex),
                builder.getStringAttr(methodInfo.name));
            return interfaceSymbolRefOp.getResult(0);
        }

        return mlir::Value();
    }

    template <typename T>
    mlir::Value mlirGenElementAccess(mlir::Location location, mlir::Value expression, mlir::Value argumentExpression, T tupleType)
    {
        // get index
        if (auto indexConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(argumentExpression.getDefiningOp()))
        {
            // this is property access
            MLIRPropertyAccessCodeLogic cl(builder, location, expression, indexConstOp.value());
            return cl.Tuple(tupleType, true);
        }
        else
        {
            llvm_unreachable("not implemented (index)");
        }
    }

    mlir::Value mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext)
    {
        auto location = loc(elementAccessExpression);

        auto expression = mlirGen(elementAccessExpression->expression.as<Expression>(), genContext);
        auto argumentExpression = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);

        auto arrayType = expression.getType();

        mlir::Type elementType;
        if (auto arrayTyped = arrayType.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            elementType = arrayTyped.getElementType();
        }
        else if (auto vectorType = arrayType.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            elementType = vectorType.getElementType();
        }
        else if (arrayType.isa<mlir_ts::StringType>())
        {
            elementType = getCharType();
        }
        else if (auto tupleType = arrayType.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            return mlirGenElementAccess(location, expression, argumentExpression, tupleType);
        }
        else if (auto tupleType = arrayType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            return mlirGenElementAccess(location, expression, argumentExpression, tupleType);
        }
        else
        {
            emitError(location) << "ElementAccessExpression: " << arrayType;
            llvm_unreachable("not implemented (ElementAccessExpression)");
        }

        auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression, argumentExpression);
        return builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);
    }

    mlir::Value mlirGen(CallExpression callExpression, const GenContext &genContext)
    {
        auto location = loc(callExpression);

        // get function ref.
        auto funcRefValue = mlirGen(callExpression->expression.as<Expression>(), genContext);
        if (!funcRefValue)
        {
            if (genContext.allowPartialResolve)
            {
                return mlir::Value();
            }

            emitError(location, "call expression is empty");

            assert(false);
            return mlir::Value();
        }

        auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
        auto definingOp = funcRefValue.getDefiningOp();
        if (funcRefValue.getType() == mlir::NoneType::get(builder.getContext()) &&
            definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
        {
            // TODO: when you resolve names such as "print", "parseInt" should return names in mlirGen(Identifier)
            auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
            auto functionName = calleeName.getValue();
            auto argumentsContext = callExpression->arguments;

            // resolve function
            MLIRCustomMethods cm(builder, location);

            SmallVector<mlir::Value, 4> operands;
            if (auto thisSymbolRefOp = funcRefValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                operands.push_back(thisSymbolRefOp.thisVal());
            }
            else if (auto thisVirtualSymbolRefOp = funcRefValue.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
            {
                operands.push_back(thisVirtualSymbolRefOp.thisVal());
            }

            if (mlir::failed(mlirGen(argumentsContext, operands, genContext)))
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "Call Method: can't resolve values of all parameters";
                }

                return mlir::Value();
            }

            return cm.callMethod(functionName, operands, genContext);
        }

        mlir::Value value;
        auto testResult = false;
        TypeSwitch<mlir::Type>(funcRefValue.getType())
            .Case<mlir::FunctionType>([&](auto calledFuncType) {
                SmallVector<mlir::Value, 4> operands;
                if (auto thisSymbolRefOp = funcRefValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
                {
                    operands.push_back(thisSymbolRefOp.thisVal());
                }
                else if (auto thisVirtualSymbolRefOp = funcRefValue.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
                {
                    operands.push_back(thisVirtualSymbolRefOp.thisVal());
                }
                else if (auto interfaceSymbolRefOp = funcRefValue.getDefiningOp<mlir_ts::InterfaceSymbolRefOp>())
                {
                    // operands.push_back(interfaceSymbolRefOp.thisRef());
                    operands.push_back(interfaceSymbolRefOp.getResult(1));
                }

                const_cast<GenContext &>(genContext).destFuncType = calledFuncType;
                if (mlir::failed(mlirGenCallOperands(location, calledFuncType, callExpression->arguments, operands, genContext)))
                {
                    if (!genContext.allowPartialResolve)
                    {
                        emitError(location) << "Call Method: can't resolve values of all parameters";
                    }
                }
                else
                {
                    // default call by name
                    auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(location, funcRefValue, operands);

                    if (calledFuncType.getNumResults() > 0)
                    {
                        value = callIndirectOp.getResult(0);
                        testResult = true;
                    }
                }

                const_cast<GenContext &>(genContext).destFuncType = nullptr;
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                // seems we are calling type constructor
                auto newOp = builder.create<mlir_ts::NewOp>(location, classType, builder.getBoolAttr(true));
                mlirGenCallConstructor(location, classType, newOp, callExpression->typeArguments, callExpression->arguments, false, true,
                                       genContext);
                value = newOp;
            })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                MLIRCodeLogic mcl(builder);
                auto refValue = mcl.GetReferenceOfLoadOp(funcRefValue);
                if (refValue)
                {
                    // seems we are calling type constructor for super()
                    mlirGenCallConstructor(location, classStorageType, refValue, callExpression->typeArguments, callExpression->arguments,
                                           true, false, genContext);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Default([&](auto type) {
                // it is not function, so just return value as maybe it has been resolved earlier like in case "<number>.ToString()"
                value = funcRefValue;
            });

        if (value)
        {
            return value;
        }

        assert(!testResult);
        return mlir::Value();
    }

    mlir::LogicalResult mlirGenCallOperands(mlir::Location location, mlir::FunctionType calledFuncType,
                                            NodeArray<Expression> argumentsContext, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        auto opArgsCount = std::distance(argumentsContext.begin(), argumentsContext.end()) + operands.size();
        auto funcArgsCount = calledFuncType.getNumInputs();

        if (mlir::failed(mlirGen(argumentsContext, operands, calledFuncType, genContext)))
        {
            return mlir::failure();
        }

        if (funcArgsCount > opArgsCount)
        {
            // -1 to exclude count params
            for (auto i = (size_t)opArgsCount; i < funcArgsCount; i++)
            {
                operands.push_back(builder.create<mlir_ts::UndefOp>(location, calledFuncType.getInput(i)));
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        for (auto expression : arguments)
        {
            auto value = mlirGen(expression, genContext);
            if (!value)
            {
                return mlir::failure();
            }

            operands.push_back(value);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, mlir::FunctionType funcType,
                                const GenContext &genContext)
    {
        auto i = operands.size(); // we need to shift in case of 'this'
        for (auto expression : arguments)
        {
            if (genContext.destFuncType)
            {
                const_cast<GenContext &>(genContext).argTypeDestFuncType = genContext.destFuncType.getInput(i);
            }

            auto value = mlirGen(expression, genContext);
            if (!value)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(loc(expression)) << "can't resolve function argument";
                }

                return mlir::failure();
            }

            if (value.getType() != funcType.getInput(i))
            {
                auto castValue = cast(loc(expression), funcType.getInput(i), value, genContext);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            const_cast<GenContext &>(genContext).argTypeDestFuncType = nullptr;

            i++;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, mlir_ts::ClassType classType, mlir::Value thisValue,
                                               NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments,
                                               bool castThisValueToClass, bool setVTable, const GenContext &genContext)
    {
        if (!classType)
        {
            return mlir::failure();
        }

        // register temp var
        auto classInfo = getClassByFullName(classType.getName().getValue());
        return mlirGenCallConstructor(location, classInfo, thisValue, typeArguments, arguments, castThisValueToClass, setVTable,
                                      genContext);
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, mlir_ts::ClassStorageType classStorageType, mlir::Value thisValue,
                                               NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments,
                                               bool castThisValueToClass, bool setVTable, const GenContext &genContext)
    {
        if (!classStorageType)
        {
            return mlir::failure();
        }

        // register temp var
        auto classInfo = getClassByFullName(classStorageType.getName().getValue());
        return mlirGenCallConstructor(location, classInfo, thisValue, typeArguments, arguments, castThisValueToClass, setVTable,
                                      genContext);
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, ClassInfo::TypePtr classInfo, mlir::Value thisValue,
                                               NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments,
                                               bool castThisValueToClass, bool setVTable, const GenContext &genContext)
    {
        assert(classInfo);

        auto virtualTable = classInfo->getHasVirtualTable();
        auto hasConstructor = classInfo->getHasConstructor();
        if (!hasConstructor && !virtualTable)
        {
            return mlir::success();
        }

        // adding call of ctor
        NodeFactory nf(NodeFactoryFlags::None);

        // to remove temp var .ctor after call
        SymbolTableScopeT varScope(symbolTable);

        auto effectiveThisValue = thisValue;
        if (castThisValueToClass)
        {
            effectiveThisValue = cast(location, classInfo->classType, thisValue, genContext);
        }

        auto varDecl = std::make_shared<VariableDeclarationDOM>(CONSTRUCTOR_TEMPVAR_NAME, classInfo->classType, location);
        declare(varDecl, effectiveThisValue);

        auto thisToken = nf.createIdentifier(S(CONSTRUCTOR_TEMPVAR_NAME));

        // set virtual table
        if (setVTable && classInfo->getHasVirtualTable())
        {
            auto _vtable_name = nf.createIdentifier(S(VTABLE_NAME));
            auto propAccess = nf.createPropertyAccessExpression(thisToken, _vtable_name);

            // set temp vtable
            auto fullClassVTableFieldName = concat(classInfo->fullName, VTABLE_NAME);
            auto vtableAddress = resolveFullNameIdentifier(location, fullClassVTableFieldName, true, genContext);
            if (!vtableAddress)
            {
                assert(genContext.allowPartialResolve);
                return mlir::failure();
            }

            auto anyTypeValue = cast(location, getAnyType(), vtableAddress, genContext);
            auto varDecl = std::make_shared<VariableDeclarationDOM>(VTABLE_NAME, anyTypeValue.getType(), location);
            declare(varDecl, anyTypeValue);

            // save vtable value
            auto setPropValue = nf.createBinaryExpression(propAccess, nf.createToken(SyntaxKind::EqualsToken), _vtable_name);

            mlirGen(setPropValue, genContext);
        }

        if (classInfo->getHasConstructor())
        {
            auto propAccess = nf.createPropertyAccessExpression(thisToken, nf.createIdentifier(S(CONSTRUCTOR_NAME)));
            auto callExpr = nf.createCallExpression(propAccess, typeArguments, arguments);

            auto callCtorValue = mlirGen(callExpr, genContext);
        }

        return mlir::success();
    }

    mlir::Value mlirGen(NewExpression newExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(newExpression);

        // 3 cases, name, index access, method call
        mlir::Type type;
        auto typeExpression = newExpression->expression;
        if (typeExpression == SyntaxKind::Identifier || typeExpression == SyntaxKind::QualifiedName ||
            typeExpression == SyntaxKind::PropertyAccessExpression)
        {
            type = getTypeByTypeName(typeExpression, genContext);

            assert(type);

            auto resultType = type;
            if (mth.isValueType(type))
            {
                resultType = getValueRefType(type);
            }

            auto newOp = builder.create<mlir_ts::NewOp>(location, resultType, builder.getBoolAttr(false));
            mlirGenCallConstructor(location, resultType.dyn_cast_or_null<mlir_ts::ClassType>(), newOp, newExpression->typeArguments,
                                   newExpression->arguments, false, true, genContext);
            return newOp;
        }
        else if (typeExpression == SyntaxKind::ElementAccessExpression)
        {
            auto elementAccessExpression = typeExpression.as<ElementAccessExpression>();
            typeExpression = elementAccessExpression->expression;
            type = getTypeByTypeName(typeExpression, genContext);

            assert(type);

            auto count = mlirGen(elementAccessExpression->argumentExpression, genContext);

            if (count.getType() != builder.getI32Type())
            {
                count = cast(location, builder.getI32Type(), count, genContext);
            }

            auto newArrOp = builder.create<mlir_ts::NewArrayOp>(location, getArrayType(type), count);
            return newArrOp;
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::LogicalResult mlirGen(DeleteExpression deleteExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(deleteExpression);

        auto expr = mlirGen(deleteExpression->expression, genContext);

        if (!expr.getType().isa<mlir_ts::RefType>() && !expr.getType().isa<mlir_ts::ValueRefType>() &&
            !expr.getType().isa<mlir_ts::ClassType>())
        {
            if (auto arrayType = expr.getType().dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                expr = cast(location, mlir_ts::RefType::get(arrayType.getElementType()), expr, genContext);
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        builder.create<mlir_ts::DeleteOp>(location, expr);

        return mlir::success();
    }

    mlir::Value mlirGen(VoidExpression voidExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(voidExpression);

        auto expr = mlirGen(voidExpression->expression, genContext);

        auto value = getUndefined(location);

        return value;
    }

    mlir::Value mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext)
    {
        auto result = mlirGen(typeOfExpression->expression, genContext);
        auto type = result.getType();
        if (type.isIntOrIndexOrFloat() && !type.isIntOrIndex())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("number")));
            return typeOfValue;
        }

        if (type == getBooleanType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getBooleanType(), getStringAttr(std::string("boolean")));
            return typeOfValue;
        }

        if (type == getStringType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("string")));
            return typeOfValue;
        }

        if (type.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("array")));
            return typeOfValue;
        }

        if (type.dyn_cast_or_null<mlir::FunctionType>())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("function")));
            return typeOfValue;
        }

        if (type == getAnyType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("object")));
            return typeOfValue;
        }

        llvm_unreachable("not implemented");
    }

    mlir::Value mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(templateExpressionAST);

        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto text = wstos(templateExpressionAST->head->rawText);
        auto head = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(text));

        // first string
        strs.push_back(head);
        for (auto span : templateExpressionAST->templateSpans)
        {
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE_EXPR(exprValue, expression)

            if (exprValue.getType() != stringType)
            {
                exprValue = cast(location, stringType, exprValue, genContext);
            }

            // expr value
            strs.push_back(exprValue);

            auto spanText = wstos(span->literal->rawText);
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            // text
            strs.push_back(spanValue);
        }

        if (strs.size() <= 1)
        {
            return head;
        }

        auto concatValues = builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return concatValues;
    }

    mlir::Value mlirGen(TaggedTemplateExpression taggedTemplateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(taggedTemplateExpressionAST);

        auto templateExpressionAST = taggedTemplateExpressionAST->_template;

        SmallVector<mlir::Attribute, 4> strs;
        SmallVector<mlir::Value, 4> vals;

        auto text = wstos(templateExpressionAST->head->rawText);

        // first string
        strs.push_back(getStringAttr(text));
        for (auto span : templateExpressionAST->templateSpans)
        {
            // expr value
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE_EXPR(exprValue, expression)

            vals.push_back(exprValue);

            auto spanText = wstos(span->literal->rawText);
            // text
            strs.push_back(getStringAttr(spanText));
        }

        // tag method
        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), strs);
        auto constStringArray = builder.create<mlir_ts::ConstantOp>(location, getConstArrayType(getStringType(), strs.size()), arrayAttr);

        auto strArrayValue = cast(location, getArrayType(getStringType()), constStringArray, genContext);

        vals.insert(vals.begin(), strArrayValue);

        auto callee = mlirGen(taggedTemplateExpressionAST->tag, genContext);

        // cast all params if needed
        auto funcType = callee.getType().cast<mlir::FunctionType>();

        SmallVector<mlir::Value, 4> operands;

        auto i = 0;
        for (auto value : vals)
        {
            if (value.getType() != funcType.getInput(i))
            {
                auto castValue = cast(value.getLoc(), funcType.getInput(i), value, genContext);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            i++;
        }

        // call
        auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(location, callee, operands);

        return callIndirectOp.getResult(0);
    }

    mlir::Value mlirGen(NullLiteral nullLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::NullOp>(loc(nullLiteral), getAnyType());
    }

    mlir::Value mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(trueLiteral), getBooleanType(), mlir::BoolAttr::get(builder.getContext(), true));
    }

    mlir::Value mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(falseLiteral), getBooleanType(), mlir::BoolAttr::get(builder.getContext(), false));
    }

    mlir::Value mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
    {
        if (numericLiteral->text.find(S(".")) == string::npos)
        {
            return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getI32Type(),
                                                       builder.getI32IntegerAttr(to_unsigned_integer(numericLiteral->text)));
        }

        return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getF32Type(),
                                                   builder.getF32FloatAttr(to_float(numericLiteral->text)));
    }

    mlir::Value mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(bigIntLiteral), builder.getI64Type(),
                                                   builder.getI64IntegerAttr(to_bignumber(bigIntLiteral->text)));
    }

    mlir::Value mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
    {
        auto text = wstos(stringLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(stringLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral, const GenContext &genContext)
    {
        auto text = wstos(noSubstitutionTemplateLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(noSubstitutionTemplateLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGenArrayLiteralExpressionNonConst(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        MLIRTypeHelper mth(builder.getContext());

        // first value
        auto isTuple = false;
        mlir::Type elementType;
        SmallVector<mlir::Type> types;
        SmallVector<mlir::Value> values;
        for (auto &item : arrayLiteral->elements)
        {
            auto itemValue = mlirGen(item, genContext);
            if (!itemValue)
            {
                // omitted expression
                continue;
            }

            auto type = itemValue.getType();

            values.push_back(itemValue);
            types.push_back(type);
            if (!elementType)
            {
                elementType = type;
            }
            else if (elementType != type)
            {
                // this is tuple.
                isTuple = true;
            }
        }

        if (isTuple)
        {
            llvm_unreachable("not implemented");
            return mlir::Value();
        }

        if (!elementType)
        {
            // in case of empty array
            llvm_unreachable("not implemented");
            return mlir::Value();
        }

        auto newArrayOp = builder.create<mlir_ts::CreateArrayOp>(loc(arrayLiteral), getArrayType(elementType), values);
        return newArrayOp;
    }

    mlir::Value mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        MLIRTypeHelper mth(builder.getContext());

        // first value
        auto isTuple = false;
        mlir::Type elementType;
        SmallVector<mlir::Type> types;
        SmallVector<mlir::Attribute> values;
        for (auto &item : arrayLiteral->elements)
        {
            auto itemValue = mlirGen(item, genContext);
            if (!itemValue)
            {
                // omitted expression
                continue;
            }

            auto constOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>();
            if (!constOp)
            {
                return mlirGenArrayLiteralExpressionNonConst(arrayLiteral, genContext);
            }

            bool copyRequired;
            auto type = mth.convertConstTypeToType(constOp.getType(), copyRequired);

            values.push_back(constOp.valueAttr());
            types.push_back(type);
            if (!elementType)
            {
                elementType = type;
            }
            else if (elementType != type)
            {
                // this is tuple.
                isTuple = true;
            }
        }

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        if (isTuple)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;
            for (auto type : types)
            {
                fieldInfos.push_back({mlir::Attribute(), type});
            }

            return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstTupleType(fieldInfos), arrayAttr);
        }

        if (!elementType)
        {
            // in case of empty array
            elementType = getAnyType();
        }

        return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstArrayType(elementType, values.size()), arrayAttr);
    }

    mlir::Value mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        // first value
        SmallVector<mlir::Type> types;
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        for (auto &item : objectLiteral->properties)
        {
            mlir::Value itemValue;
            mlir::Attribute fieldId;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                itemValue = mlirGen(propertyAssignment->initializer, genContext);
                auto name = MLIRHelper::getName(propertyAssignment->name);
                if (name.empty())
                {
                    auto value = mlirGen(propertyAssignment->name.as<Expression>(), genContext);
                    fieldId = mcl.ExtractAttr(value);
                }
                else
                {
                    auto namePtr = StringRef(name).copy(stringAllocator);
                    fieldId = mcl.TupleFieldName(namePtr);
                }
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                itemValue = mlirGen(shorthandPropertyAssignment->name.as<Expression>(), genContext);
                auto name = MLIRHelper::getName(shorthandPropertyAssignment->name);
                auto namePtr = StringRef(name).copy(stringAllocator);
                fieldId = mcl.TupleFieldName(namePtr);
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1)");
            }

            assert(itemValue);

            mlir::Type type;
            mlir::Attribute value;
            if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(itemValue.getDefiningOp()))
            {
                value = constOp.valueAttr();
                type = constOp.getType();
            }
            else if (auto symRefOp = dyn_cast_or_null<mlir_ts::SymbolRefOp>(itemValue.getDefiningOp()))
            {
                value = symRefOp.identifierAttr();
                type = symRefOp.getType();
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1), must be const object or global symbol");
                continue;
            }

            values.push_back(value);
            types.push_back(type);
            fieldInfos.push_back({fieldId, type});
        }

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        return builder.create<mlir_ts::ConstantOp>(loc(objectLiteral), getConstTupleType(fieldInfos), arrayAttr);
    }

    mlir::Value mlirGen(Identifier identifier, const GenContext &genContext)
    {
        auto location = loc(identifier);

        // resolve name
        auto name = MLIRHelper::getName(identifier);

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
            // begin of logic: outer vars
            auto valueRegion = value.first.getParentRegion();
            auto isOuterVar = false;
            if (genContext.funcOp)
            {
                auto funcRegion = const_cast<GenContext &>(genContext).funcOp.getCallableRegion();
                isOuterVar = !funcRegion->isAncestor(valueRegion);
            }

            // auto isOuterFunctionScope = value.second->getFuncOp() != genContext.funcOp;
            if (isOuterVar && genContext.passResult)
            {
                LLVM_DEBUG(dbgs() << "\n...capturing var: [" << value.second->getName() << "] value pair: " << value.first << " type: "
                                  << value.second->getType() << " readwrite: " << value.second->getReadWriteAccess() << "\n\n";);

                genContext.passResult->outerVariables.insert({value.second->getName(), value.second});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            LLVM_DEBUG(dbgs() << "??? variable: " << name << " type: " << value.first.getType() << "\n");

            // load value if memref
            auto valueType = value.first.getType().cast<mlir_ts::RefType>().getElementType();
            return builder.create<mlir_ts::LoadOp>(value.first.getLoc(), valueType, value.first);
        }

        return mlir::Value();
    }

    mlir::Value resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir::FunctionType funcType,
                                           const GenContext &genContext)
    {
        // check if required capture of vars
        auto captureVars = getCaptureVarsMap().find(name);
        if (captureVars != getCaptureVarsMap().end())
        {
            auto newFuncType = builder.getFunctionType(funcType.getInputs().slice(1), funcType.getResults());

            auto funcSymbolOp =
                builder.create<mlir_ts::SymbolRefOp>(location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));

            MLIRCodeLogic mcl(builder);
            SmallVector<mlir::Value> capturedValues;
            for (auto &item : captureVars->getValue())
            {
                auto varValue = mlirGen(location, item.first(), genContext);

                // review capturing by ref.  it should match storage type
                auto refValue = mcl.GetReferenceOfLoadOp(varValue);
                if (refValue)
                {
                    capturedValues.push_back(refValue);
                    // set var as captures
                    if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                    {
                        varOp.capturedAttr(builder.getBoolAttr(true));
                    }
                }
                else
                {
                    // this is not ref, this is const value
                    capturedValues.push_back(varValue);
                }
            }

            // add attributes to track which one sent by ref.
            auto captured = builder.create<mlir_ts::CaptureOp>(location, funcType.getInput(0), capturedValues);
            return builder.create<mlir_ts::TrampolineOp>(location, newFuncType, funcSymbolOp, captured);
        }

        return mlir::Value();
    }

    mlir::Value resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto funcOp = fn->getValue();
            auto funcType = funcOp.getType();

            if (auto trampOp = resolveFunctionWithCapture(location, name, funcType, genContext))
            {
                return trampOp;
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, funcType,
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
            return symbOp;
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
            return builder.create<mlir_ts::ConstantOp>(location, getEnumType(enumTypeInfo.first), enumTypeInfo.second);
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            if (!classInfo->classType)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "can't find class: " << name << "\n";
                }

                return mlir::Value();
            }

            return builder.create<mlir_ts::ClassRefOp>(
                location, classInfo->classType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));
        }

        if (getInterfacesMap().count(name))
        {
            auto interfaceInfo = getInterfacesMap().lookup(name);
            if (!interfaceInfo->interfaceType)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "can't find interface: " << name << "\n";
                }

                return mlir::Value();
            }

            return builder.create<mlir_ts::InterfaceRefOp>(
                location, interfaceInfo->interfaceType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceInfo->interfaceType.getName().getValue()));
        }

        if (getTypeAliasMap().count(name))
        {
            auto typeAliasInfo = getTypeAliasMap().lookup(name);
            return builder.create<mlir_ts::TypeRefOp>(location, typeAliasInfo);
        }

        if (getNamespaceMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            assert(namespaceInfo);
            return builder.create<mlir_ts::NamespaceRefOp>(location,
                                                           mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName));
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto namespaceInfo = getNamespaceByFullName(fullName);
            if (namespaceInfo)
            {
                assert(namespaceInfo);
                return builder.create<mlir_ts::NamespaceRefOp>(location,
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName));
            }

            auto classInfo = getClassByFullName(fullName);
            if (classInfo)
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));
            }

            auto interfaceInfo = getInterfaceByFullName(fullName);
            if (interfaceInfo)
            {
                return builder.create<mlir_ts::InterfaceRefOp>(
                    location, interfaceInfo->interfaceType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceInfo->interfaceType.getName().getValue()));
            }

            assert(false);
        }

        return mlir::Value();
    }

    mlir::Value resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess, const GenContext &genContext)
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
        if (!value->getReadWriteAccess() && value->getType().isa<mlir_ts::StringType>())
        {
            // load address of const object in global
            return builder.create<mlir_ts::AddressOfConstStringOp>(location, value->getType(), value->getName());
        }
        else
        {
            auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(value->getType()), value->getName());
            if (asAddess)
            {
                return address;
            }

            return builder.create<mlir_ts::LoadOp>(location, value->getType(), address);
        }
    }

    mlir::Value resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // built in types
        if (name == UNDEFINED_NAME)
        {
            return getUndefined(location);
        }

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

        // search in root namespace
        auto saveNamespace = currentNamespace;
        currentNamespace = rootNamespace;
        value = resolveIdentifierInNamespace(location, name, genContext);
        currentNamespace = saveNamespace;
        if (value)
        {
            return value;
        }

        // try to resolve 'this' if not resolved yet
        if (genContext.thisType && name == THIS_NAME)
        {
            return builder.create<mlir_ts::ClassRefOp>(
                location, genContext.thisType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genContext.thisType.cast<mlir_ts::ClassType>().getName().getValue()));
        }

        if (genContext.thisType && name == SUPER_NAME)
        {
            auto thisValue = mlirGen(location, THIS_NAME, genContext);

            auto classInfo = getClassByFullName(genContext.thisType.cast<mlir_ts::ClassType>().getName().getValue());
            auto baseClassInfo = classInfo->baseClasses.front();

            return mlirGenPropertyAccessExpression(location, thisValue, baseClassInfo->fullName, genContext);
        }

        value = resolveFullNameIdentifier(location, name, false, genContext);
        if (value)
        {
            return value;
        }

        return mlir::Value();
    }

    mlir::Value mlirGen(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifier(location, name, genContext);
        if (value)
        {
            return value;
        }

        // unresolved reference (for call for example)
        // TODO: put assert here to see which ref names are not resolved
        return builder.create<mlir_ts::UnresolvedSymbolRefOp>(location, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    mlir::LogicalResult mlirGen(TypeAliasDeclaration typeAliasDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(typeAliasDeclarationAST->name);
        if (!name.empty())
        {
            auto type = getType(typeAliasDeclarationAST->type);
            getTypeAliasMap().insert({name, type});
            return mlir::success();
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::Value mlirGenModuleReference(Node moduleReference, const GenContext &genContext)
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
            auto value = mlirGenModuleReference(importEqualsDeclarationAST->moduleReference, genContext);
            if (auto namespaceOp = value.getDefiningOp<mlir_ts::NamespaceRefOp>())
            {
                getImportEqualsMap().insert({name, namespaceOp.identifier()});
                return mlir::success();
            }
            else if (auto classRefOp = value.getDefiningOp<mlir_ts::ClassRefOp>())
            {
                getImportEqualsMap().insert({name, classRefOp.identifier()});
                return mlir::success();
            }
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(enumDeclarationAST->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::failure();
        }

        auto namePtr = StringRef(name).copy(stringAllocator);

        SmallVector<mlir::NamedAttribute> enumValues;
        int64_t index = 0;
        auto activeBits = 0;
        for (auto enumMember : enumDeclarationAST->members)
        {
            auto memberName = MLIRHelper::getName(enumMember->name);
            if (memberName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            mlir::Attribute enumValueAttr;
            if (enumMember->initializer)
            {
                GenContext enumValueGenContext(genContext);
                enumValueGenContext.allowConstEval = true;
                auto enumValue = mlirGen(enumMember->initializer, enumValueGenContext);
                if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(enumValue.getDefiningOp()))
                {
                    enumValueAttr = constOp.valueAttr();
                    if (auto intAttr = enumValueAttr.dyn_cast_or_null<mlir::IntegerAttr>())
                    {
                        index = intAttr.getInt();
                        auto currentActiveBits = (int)intAttr.getValue().getActiveBits();
                        if (currentActiveBits > activeBits)
                        {
                            activeBits = currentActiveBits;
                        }
                    }
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
            else
            {
                enumValueAttr = builder.getI32IntegerAttr(index);
            }

            enumValues.push_back({mlir::Identifier::get(memberName, builder.getContext()), enumValueAttr});
            index++;
        }

        // count used bits
        auto indexUsingBits = std::floor(std::log2(index)) + 1;
        if (indexUsingBits > activeBits)
        {
            activeBits = indexUsingBits;
        }

        // get type by size
        auto bits = 32;
        if (bits < activeBits)
        {
            bits = 64;
            if (bits < activeBits)
            {
                bits = 128;
            }
        }

        auto enumIntType = builder.getIntegerType(bits);
        SmallVector<mlir::NamedAttribute> adjustedEnumValues;
        for (auto enumItem : enumValues)
        {
            if (auto intAttr = enumItem.second.dyn_cast_or_null<mlir::IntegerAttr>())
            {
                adjustedEnumValues.push_back({enumItem.first, mlir::IntegerAttr::get(enumIntType, intAttr.getInt())});
            }
            else
            {
                adjustedEnumValues.push_back(enumItem);
            }
        }

        getEnumsMap().insert({namePtr, std::make_pair(enumIntType, mlir::DictionaryAttr::get(builder.getContext(), adjustedEnumValues))});

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(classDeclarationAST);

        auto declareClass = false;
        auto newClassPtr = mlirGenClassInfo(classDeclarationAST, declareClass, genContext);
        if (!newClassPtr)
        {
            return mlir::failure();
        }

        if (mlir::failed(mlirGenClassStorageType(classDeclarationAST, newClassPtr, declareClass, genContext)))
        {
            return mlir::failure();
        }

        // clear all flags
        for (auto &classMember : classDeclarationAST->members)
        {
            classMember->processed = false;
        }

        mlirGenClassDefaultConstructor(classDeclarationAST, newClassPtr, genContext);

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            for (auto &classMember : classDeclarationAST->members)
            {
                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, declareClass, genContext)))
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

        // generate vtable for interfaces
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(
                    mlirGenClassHeritageClauseImplements(classDeclarationAST, newClassPtr, heritageClause, declareClass, genContext)))
            {
                return mlir::failure();
            }
        }

        mlirGenClassVirtualTableDefinition(location, newClassPtr, genContext);

        return mlir::success();
    }

    ClassInfo::TypePtr mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, bool &declareClass, const GenContext &genContext)
    {
        declareClass = false;

        auto name = MLIRHelper::getName(classDeclarationAST->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return ClassInfo::TypePtr();
        }

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        ClassInfo::TypePtr newClassPtr;
        if (fullNameClassesMap.count(fullNamePtr))
        {
            newClassPtr = fullNameClassesMap.lookup(fullNamePtr);
            getClassesMap().insert({namePtr, newClassPtr});
            declareClass = !newClassPtr->classType;
        }
        else
        {
            // register class
            newClassPtr = std::make_shared<ClassInfo>();
            newClassPtr->name = namePtr;
            newClassPtr->fullName = fullNamePtr;
            newClassPtr->isAbstract = hasModifier(classDeclarationAST, SyntaxKind::AbstractKeyword);
            newClassPtr->hasVirtualTable = newClassPtr->isAbstract;

            getClassesMap().insert({namePtr, newClassPtr});
            fullNameClassesMap.insert(fullNamePtr, newClassPtr);
            declareClass = true;
        }

        return newClassPtr;
    }

    mlir::LogicalResult mlirGenClassStorageType(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr, bool declareClass,
                                                const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // add base classes
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(
                    mlirGenClassHeritageClause(classDeclarationAST, newClassPtr, heritageClause, fieldInfos, declareClass, genContext)))
            {
                return mlir::failure();
            }
        }

        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(classDeclarationAST, newClassPtr, classMember, fieldInfos, declareClass, genContext)))
            {
                return mlir::failure();
            }
        }

        if (declareClass)
        {
            if (newClassPtr->getHasVirtualTableVariable())
            {
                MLIRCodeLogic mcl(builder);
                auto fieldId = mcl.TupleFieldName(VTABLE_NAME);
                fieldInfos.insert(fieldInfos.begin(), {fieldId, getAnyType()});
            }

            auto classFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), newClassPtr->fullName);
            newClassPtr->classType = getClassType(classFullNameSymbol, getClassStorageType(classFullNameSymbol, fieldInfos));
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassHeritageClause(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                   HeritageClause heritageClause, SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                   bool declareClass, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);

        if (heritageClause->token == SyntaxKind::ExtendsKeyword)
        {
            auto &baseClassInfos = newClassPtr->baseClasses;

            for (auto &extendingType : heritageClause->types)
            {
                auto baseType = mlirGen(extendingType->expression, genContext);
                TypeSwitch<mlir::Type>(baseType.getType())
                    .template Case<mlir_ts::ClassType>([&](auto classType) {
                        auto classInfo = getClassByFullName(classType.getName().getValue());

                        auto baseName = classType.getName().getValue();
                        auto fieldId = mcl.TupleFieldName(baseName);
                        fieldInfos.push_back({fieldId, classType.getStorageType()});

                        baseClassInfos.push_back(classInfo);
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
                if (implementingType->processed)
                {
                    continue;
                }

                auto ifaceType = mlirGen(implementingType->expression, genContext);
                TypeSwitch<mlir::Type>(ifaceType.getType())
                    .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                        auto interfaceInfo = getInterfaceByFullName(interfaceType.getName().getValue());
                        interfaceInfos.push_back({interfaceInfo, -1});
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassFieldMember(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                ClassElement classMember, SmallVector<mlir_ts::FieldInfo> &fieldInfos, bool declareClass,
                                                const GenContext &genContext)
    {
        auto location = loc(classMember);

        MLIRCodeLogic mcl(builder);

        auto &staticFieldInfos = newClassPtr->staticFields;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        if (classMember == SyntaxKind::Constructor)
        {
            newClassPtr->hasConstructor = true;
        }

        auto isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        if (isAbstract)
        {
            newClassPtr->hasVirtualTable = true;
        }

        auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
        if (!isStatic && !declareClass)
        {
            return mlir::success();
        }

        auto noneType = mlir::NoneType::get(builder.getContext());
        if (classMember == SyntaxKind::PropertyDeclaration)
        {
            // property declaration
            auto propertyDeclaration = classMember.as<PropertyDeclaration>();

            auto memberName = MLIRHelper::getName(propertyDeclaration->name);
            if (memberName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            memberNamePtr = StringRef(memberName).copy(stringAllocator);
            fieldId = mcl.TupleFieldName(memberNamePtr);

            if (!isStatic)
            {
                auto typeAndInit = getTypeAndInit(propertyDeclaration, genContext);
                type = typeAndInit.first;
                if (typeAndInit.second)
                {
                    newClassPtr->hasInitializers = true;
                }

                LLVM_DEBUG(dbgs() << "\n+++ class field: " << fieldId << " type: " << type << "\n\n");

                if (!type || type == noneType)
                {
                    return mlir::failure();
                }

                fieldInfos.push_back({fieldId, type});
            }
            else
            {
                // register global
                auto fullClassStaticFieldName = concat(newClassPtr->fullName, memberNamePtr);
                registerVariable(
                    location, fullClassStaticFieldName, true, VariableClass::Var,
                    [&]() { return getTypeAndInit(propertyDeclaration, genContext); }, genContext);

                if (declareClass)
                {
                    staticFieldInfos.push_back({fieldId, fullClassStaticFieldName});
                }
            }
        }

        if (classMember == SyntaxKind::Constructor && !isStatic)
        {
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

                auto parameterName = MLIRHelper::getName(parameter->name);
                if (parameterName.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                memberNamePtr = StringRef(parameterName).copy(stringAllocator);
                fieldId = mcl.TupleFieldName(memberNamePtr);

                auto typeAndInit = getTypeAndInit(parameter, genContext);
                type = typeAndInit.first;

                LLVM_DEBUG(dbgs() << "\n+++ class auto-gen field: " << fieldId << " type: " << type << "\n\n");
                if (!type || type == noneType)
                {
                    return mlir::failure();
                }

                fieldInfos.push_back({fieldId, type});
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                       const GenContext &genContext)
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
            auto generatedConstructor = nf.createConstructorDeclaration(undefined, undefined, undefined, body);
            classDeclarationAST->members.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());

        MethodInfo emptyMethod;
        // TODO: ...
        llvm::SmallVector<MethodInfo> virtualTable;
        auto result = newInterfacePtr->getVirtualTable(virtualTable, [&](std::string name, mlir::FunctionType funcType) -> MethodInfo & {
            auto index = newClassPtr->getMethodIndex(name);
            if (index >= 0)
            {
                auto &foundMethod = newClassPtr->methods[index];
                auto foundMethodFunctionType = foundMethod.funcOp.getType().cast<mlir::FunctionType>();

                auto result = mth.TestFunctionTypesMatch(funcType, foundMethodFunctionType, 1);
                if (result.result != MatchResultType::Match)
                {
                    emitError(location) << "method signature not matching for '" << name << "' for interface '" << newInterfacePtr->fullName
                                        << "' in class '" << newClassPtr->fullName << "'";

                    return emptyMethod;
                }

                return foundMethod;
            }

            emitError(location) << "can't find method '" << name << "' for interface '" << newInterfacePtr->fullName << "' in class '"
                                << newClassPtr->fullName << "'";

            return emptyMethod;
        });

        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = concat(newClassPtr->fullName, newInterfacePtr->fullName, VTABLE_NAME);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableClass::Var,
            [&]() {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto method : virtualTable)
                {
                    auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                        location, method.funcOp.getType(), mlir::FlatSymbolRefAttr::get(builder.getContext(), method.funcOp.sym_name()));

                    vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                        location, virtTuple, methodConstName, vtableValue, builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex++)));
                }

                return std::pair<mlir::Type, mlir::Value>{virtTuple, vtableValue};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                             HeritageClause heritageClause, bool declareClass, const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ImplementsKeyword)
        {
            return mlir::success();
        }

        // TODO: finish method.

        for (auto &implementingType : heritageClause->types)
        {
            if (implementingType->processed)
            {
                continue;
            }

            auto ifaceType = mlirGen(implementingType->expression, genContext);
            auto success = false;
            TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceByFullName(interfaceType.getName().getValue());
                    success = !failed(
                        mlirGenClassVirtualTableDefinitionForInterface(loc(implementingType), newClassPtr, interfaceInfo, genContext));
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<MethodInfo> &virtualTable)
    {
        MLIRCodeLogic mcl(builder);

        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            fields.push_back({mcl.TupleFieldName(vtableRecord.name), vtableRecord.funcOp.getType()});
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &virtualTable)
    {
        MLIRCodeLogic mcl(builder);

        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isInterfaceVTable)
            {
                fields.push_back({mcl.TupleFieldName(vtableRecord.methodInfo.name), getAnyType()});
            }
            else
            {
                fields.push_back({mcl.TupleFieldName(vtableRecord.methodInfo.name), vtableRecord.methodInfo.funcOp.getType()});
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

        MLIRTypeHelper mth(builder.getContext());

        // register global
        auto fullClassVTableFieldName = concat(newClassPtr->fullName, VTABLE_NAME);
        registerVariable(
            location, fullClassVTableFieldName, true, VariableClass::Var,
            [&]() {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto vtRecord : virtualTable)
                {
                    if (vtRecord.isInterfaceVTable)
                    {
                        // TODO: write correct full name for vtable
                        auto fullClassInterfaceVTableFieldName = concat(newClassPtr->fullName, vtRecord.methodInfo.name, VTABLE_NAME);
                        auto interfaceVTableValue =
                            resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
                        assert(interfaceVTableValue);

                        auto interfaceVTableValueAsAny = cast(location, getAnyType(), interfaceVTableValue, genContext);

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, interfaceVTableValueAsAny, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex++)));
                    }
                    else
                    {
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, vtRecord.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), vtRecord.methodInfo.funcOp.sym_name()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex++)));
                    }
                }

                return std::pair<mlir::Type, mlir::Value>{virtTuple, vtableValue};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                 ClassElement classMember, bool declareClass, const GenContext &genContext)
    {
        if (classMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(classMember);

        auto &methodInfos = newClassPtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
        auto isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        auto isConstructor = classMember == SyntaxKind::Constructor;
        if (classMember == SyntaxKind::MethodDeclaration || isConstructor || classMember == SyntaxKind::GetAccessor ||
            classMember == SyntaxKind::SetAccessor)
        {
            auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
            std::string methodName;
            std::string propertyName;
            getMethodNameOrPropertyName(funcLikeDeclaration, methodName, propertyName);

            if (methodName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            classMember->parent = classDeclarationAST;

            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = newClassPtr->classType;
            funcGenContext.passResult = nullptr;
            if (isConstructor)
            {
                // adding missing statements
                generateConstructorStatements(classDeclarationAST, funcGenContext);
            }

            auto funcOp = mlirGenFunctionLikeDeclaration(funcLikeDeclaration, funcGenContext);

            // clean up
            const_cast<GenContext &>(genContext).generatedStatements.clear();

            if (!funcOp)
            {
                return mlir::failure();
            }

            funcLikeDeclaration->processed = true;

            if (declareClass)
            {
                methodInfos.push_back({methodName, funcOp, isStatic, isAbstract});
                addAccessor(newClassPtr, classMember, propertyName, funcOp, isStatic, isAbstract);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        for (auto &classMember : classDeclarationAST->members)
        {
            auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                if (isStatic)
                {
                    continue;
                }

                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (!propertyDeclaration->initializer)
                {
                    continue;
                }

                auto memberName = MLIRHelper::getName(propertyDeclaration->name);
                if (memberName.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                auto memberNamePtr = StringRef(memberName).copy(stringAllocator);

                auto _this = nf.createIdentifier(S(THIS_NAME));
                auto _name = nf.createIdentifier(stows(std::string(memberNamePtr)));
                auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                auto _this_name_equal =
                    nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), propertyDeclaration->initializer);
                auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
            }

            if (classMember == SyntaxKind::Constructor)
            {
                if (isStatic)
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

                    auto propertyName = MLIRHelper::getName(parameter->name);
                    if (propertyName.empty())
                    {
                        llvm_unreachable("not implemented");
                        return mlir::failure();
                    }

                    auto propertyNamePtr = StringRef(propertyName).copy(stringAllocator);

                    auto _this = nf.createIdentifier(stows(THIS_NAME));
                    auto _name = nf.createIdentifier(stows(std::string(propertyNamePtr)));
                    auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                    auto _this_name_equal = nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), _name);
                    auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                    const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
                }
            }
        }

        return mlir::success();
    }

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(InterfaceDeclaration interfaceDeclarationAST, bool &declareInterface,
                                                const GenContext &genContext)
    {
        declareInterface = false;

        auto name = MLIRHelper::getName(interfaceDeclarationAST->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return InterfaceInfo::TypePtr();
        }

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        InterfaceInfo::TypePtr newInterfacePtr;
        if (fullNameClassesMap.count(fullNamePtr))
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

            getInterfacesMap().insert({namePtr, newInterfacePtr});
            fullNameInterfacesMap.insert(fullNamePtr, newInterfacePtr);
            declareInterface = true;
        }

        return newInterfacePtr;
    }

    mlir::LogicalResult mlirGen(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(interfaceDeclarationAST);

        auto declareInterface = false;
        auto newInterfacePtr = mlirGenInterfaceInfo(interfaceDeclarationAST, declareInterface, genContext);
        if (!newInterfacePtr)
        {
            return mlir::failure();
        }

        if (mlir::failed(mlirGenInterfaceType(interfaceDeclarationAST, newInterfacePtr, declareInterface, genContext)))
        {
            return mlir::failure();
        }

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
                if (mlir::failed(mlirGenInterfaceMethodMember(interfaceDeclarationAST, newInterfacePtr, interfaceMember, declareInterface,
                                                              genContext)))
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

        return mlir::success();
    }

    mlir::LogicalResult mlirGenInterfaceType(InterfaceDeclaration interfaceDeclarationAST, InterfaceInfo::TypePtr newInterfacePtr,
                                             bool declareInterface, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        if (newInterfacePtr)
        {
            /*
            // add virtual table field
            MLIRCodeLogic mcl(builder);
            auto vtFieldId = mcl.TupleFieldName(VTABLE_NAME);
            fieldInfos.insert(fieldInfos.begin(), {vtFieldId, getAnyType()});

            auto thisFieldId = mcl.TupleFieldName(THIS_NAME);
            fieldInfos.insert(fieldInfos.begin(), {thisFieldId, getAnyType()});
            */

            auto interfaceFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), newInterfacePtr->fullName);
            newInterfacePtr->interfaceType = getInterfaceType(interfaceFullNameSymbol /*, fieldInfos*/);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenInterfaceMethodMember(InterfaceDeclaration interfaceDeclarationAST, InterfaceInfo::TypePtr newInterfacePtr,
                                                     TypeElement interfaceMember, bool declareInterface, const GenContext &genContext)
    {
        if (interfaceMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(interfaceMember);

        auto &fieldInfos = newInterfacePtr->fields;
        auto &methodInfos = newInterfacePtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        MLIRCodeLogic mcl(builder);
        auto noneType = mlir::NoneType::get(builder.getContext());

        if (declareInterface && interfaceMember == SyntaxKind::PropertySignature)
        {
            // property declaration
            auto propertyDeclaration = interfaceMember.as<PropertySignature>();

            auto memberName = MLIRHelper::getName(propertyDeclaration->name);
            if (memberName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            memberNamePtr = StringRef(memberName).copy(stringAllocator);
            fieldId = mcl.TupleFieldName(memberNamePtr);

            auto typeAndInit = getTypeAndInit(propertyDeclaration, genContext);
            type = typeAndInit.first;

            LLVM_DEBUG(dbgs() << "\n+++ interface field: " << fieldId << " type: " << type << "\n\n");

            if (!type || type == noneType)
            {
                return mlir::failure();
            }

            fieldInfos.push_back({fieldId, type});
        }

        if (interfaceMember == SyntaxKind::MethodSignature)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            std::string methodName;
            std::string propertyName;
            getMethodNameOrPropertyName(methodSignature, methodName, propertyName);

            if (methodName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            interfaceMember->parent = interfaceDeclarationAST;

            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = newInterfacePtr->interfaceType;
            funcGenContext.passResult = nullptr;

            auto res = mlirGenFunctionSignaturePrototype(methodSignature, funcGenContext);
            auto funcType = std::get<1>(res);

            if (!funcType)
            {
                return mlir::failure();
            }

            methodSignature->processed = true;

            if (declareInterface)
            {
                methodInfos.push_back({methodName, funcType, (int)methodInfos.size()});
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult getMethodNameOrPropertyName(SignatureDeclarationBase methodSignature, std::string &methodName,
                                                    std::string &propertyName)
    {
        if (methodSignature == SyntaxKind::Constructor)
        {
            methodName = std::string(CONSTRUCTOR_NAME);
        }
        else if (methodSignature == SyntaxKind::GetAccessor)
        {
            propertyName = MLIRHelper::getName(methodSignature->name);
            methodName = std::string("get_") + propertyName;
        }
        else if (methodSignature == SyntaxKind::SetAccessor)
        {
            propertyName = MLIRHelper::getName(methodSignature->name);
            methodName = std::string("set_") + propertyName;
        }
        else
        {
            methodName = MLIRHelper::getName(methodSignature->name);
        }

        return mlir::success();
    }

    void addAccessor(ClassInfo::TypePtr newClassPtr, ClassElement classMember, std::string &propertyName, mlir_ts::FuncOp funcOp,
                     bool isStatic, bool isVirtual)
    {
        auto &accessorInfos = newClassPtr->accessors;

        auto accessorIndex = newClassPtr->getAccessorIndex(propertyName);
        if (accessorIndex < 0)
        {
            accessorInfos.push_back({propertyName, {}, {}, isStatic, isVirtual});
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

    mlir::Value cast(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        // class to string
        if (auto stringType = type.dyn_cast_or_null<mlir_ts::StringType>())
        {
            if (auto classType = value.getType().dyn_cast_or_null<mlir_ts::ClassType>())
            {
                auto className = classType.getName().getValue();
                auto fullToStringName = className + ".toString";
                auto callRes =
                    builder.create<mlir_ts::CallOp>(location, fullToStringName.str(), mlir::TypeRange{stringType}, mlir::ValueRange{value});
                return callRes.getResult(0);
            }
        }

        // class to interface
        if (auto interfaceType = type.dyn_cast_or_null<mlir_ts::InterfaceType>())
        {
            if (auto classType = value.getType().dyn_cast_or_null<mlir_ts::ClassType>())
            {
                auto vtableAccess = mlirGenPropertyAccessExpression(location, value, VTABLE_NAME, genContext);

                auto classInfo = getClassByFullName(classType.getName().getValue());
                assert(classInfo);

                auto implementIndex = classInfo->getImplementIndex(interfaceType.getName().getValue());
                if (implementIndex >= 0)
                {
                    auto interfaceVirtTableIndex = classInfo->implements[implementIndex].virtualIndex;

                    auto interfaceVTablePtr =
                        builder.create<mlir_ts::VTableOffsetRefOp>(location, getAnyType(), vtableAccess, interfaceVirtTableIndex);

                    auto newInterface =
                        builder.create<mlir_ts::NewInterfaceOp>(location, mlir::TypeRange{interfaceType}, value, interfaceVTablePtr);
                    return newInterface;
                }

                assert(false);

                return mlir::Value();
            }
        }

        return builder.create<mlir_ts::CastOp>(location, type, value);
    }

    mlir::Type getType(Node typeReferenceAST)
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
            return getFunctionType(typeReferenceAST.as<FunctionTypeNode>());
        }
        else if (kind == SyntaxKind::TupleType)
        {
            return getTupleType(typeReferenceAST.as<TupleTypeNode>());
        }
        else if (kind == SyntaxKind::TypeLiteral)
        {
            return getTupleType(typeReferenceAST.as<TypeLiteralNode>());
        }
        else if (kind == SyntaxKind::ArrayType)
        {
            return getArrayType(typeReferenceAST.as<ArrayTypeNode>());
        }
        else if (kind == SyntaxKind::UnionType)
        {
            return getUnionType(typeReferenceAST.as<UnionTypeNode>());
        }
        else if (kind == SyntaxKind::IntersectionType)
        {
            return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>());
        }
        else if (kind == SyntaxKind::ParenthesizedType)
        {
            return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>());
        }
        else if (kind == SyntaxKind::LiteralType)
        {
            return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
        }
        else if (kind == SyntaxKind::TypeReference)
        {
            GenContext genContext;
            return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeQuery)
        {
            GenContext genContext;
            return getTypeByTypeQuery(typeReferenceAST.as<TypeQueryNode>(), genContext);
        }
        else if (kind == SyntaxKind::AnyKeyword)
        {
            return getAnyType();
        }

        llvm_unreachable("not implemented type declaration");
        // return getAnyType();
    }

    mlir::Type getTypeByTypeName(Node node, const GenContext &genContext)
    {
        mlir::Value value;
        if (node == SyntaxKind::QualifiedName)
        {
            value = mlirGen(node.as<QualifiedName>(), genContext);
        }
        else
        {
            value = mlirGen(node.as<Expression>(), genContext);
        }

        if (value)
        {
            auto type = value.getType();

            // extra code for extracting enum storage type
            // TODO: think if you can avoid doing it
            if (auto enumType = type.dyn_cast_or_null<mlir_ts::EnumType>())
            {
                return enumType.getElementType();
            }

            assert(type);

            return type;
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        return getTypeByTypeName(typeReferenceAST->typeName, genContext);
    }

    mlir::Type getTypeByTypeQuery(TypeQueryNode typeQueryAST, const GenContext &genContext)
    {
        auto value = mlirGen(typeQueryAST->exprName.as<Expression>(), genContext);
        assert(value);

        LLVM_DEBUG(dbgs() << "typeQuery: "; value.getType().dump(); dbgs() << "\n";);

        return value.getType();
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
        return getEnumType(builder.getI32Type());
    }

    mlir_ts::EnumType getEnumType(mlir::Type elementType)
    {
        return mlir_ts::EnumType::get(elementType);
    }

    mlir_ts::ClassStorageType getClassStorageType(mlir::FlatSymbolRefAttr name, mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ClassStorageType::get(builder.getContext(), name, fieldInfos);
    }

    mlir_ts::ClassType getClassType(mlir::FlatSymbolRefAttr name, mlir::Type storageType)
    {
        return mlir_ts::ClassType::get(name, storageType);
    }

    mlir_ts::InterfaceType getInterfaceType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::InterfaceType::get(name);
    }

    mlir_ts::ConstArrayType getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size)
    {
        auto type = getType(arrayTypeAST->elementType);
        return getConstArrayType(type, size);
    }

    mlir_ts::ConstArrayType getConstArrayType(mlir::Type elementType, unsigned size)
    {
        assert(elementType);
        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir_ts::ArrayType getArrayType(ArrayTypeNode arrayTypeAST)
    {
        auto type = getType(arrayTypeAST->elementType);
        return getArrayType(type);
    }

    mlir_ts::ArrayType getArrayType(mlir::Type elementType)
    {
        return mlir_ts::ArrayType::get(elementType);
    }

    mlir_ts::ValueRefType getValueRefType(mlir::Type elementType)
    {
        return mlir_ts::ValueRefType::get(elementType);
    }

    mlir::Value getUndefined(mlir::Location location)
    {
        return builder.create<mlir_ts::UndefOp>(location, getOptionalType(getUndefPlaceHolderType()));
    }

    void getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types)
    {
        MLIRCodeLogic mcl(builder);
        mlir::Attribute attrVal;
        for (auto typeItem : tupleType->elements)
        {
            if (typeItem == SyntaxKind::NamedTupleMember)
            {
                auto namedTupleMember = typeItem.as<NamedTupleMember>();
                auto namePtr = MLIRHelper::getName(namedTupleMember->name, stringAllocator);

                auto type = getType(namedTupleMember->type);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else if (typeItem == SyntaxKind::LiteralType)
            {
                auto literalTypeNode = typeItem.as<LiteralTypeNode>();
                GenContext genContext;
                auto literalValue = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
                auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(literalValue.getDefiningOp());
                attrVal = constantOp.valueAttr();
                continue;
            }
            else
            {
                auto type = getType(typeItem);

                assert(type);
                types.push_back({attrVal, type});
            }

            attrVal = mlir::Attribute();
        }
    }

    void getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types)
    {
        MLIRCodeLogic mcl(builder);
        for (auto typeItem : typeLiteral->members)
        {
            if (typeItem == SyntaxKind::PropertySignature)
            {
                auto propertySignature = typeItem.as<PropertySignature>();
                auto namePtr = MLIRHelper::getName(propertySignature->name, stringAllocator);

                auto type = getType(propertySignature->type);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else
            {
                auto type = getType(typeItem);

                assert(type);
                types.push_back({mlir::Attribute(), type});
            }
        }
    }

    mlir_ts::ConstTupleType getConstTupleType(TupleTypeNode tupleType)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types);
        return getConstTupleType(types);
    }

    mlir_ts::ConstTupleType getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ConstTupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::TupleType getTupleType(TupleTypeNode tupleType)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(TypeLiteralNode typeLiteral)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(typeLiteral, types);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    mlir::FunctionType getFunctionType(FunctionTypeNode functionType)
    {
        auto resultType = getType(functionType->type);
        SmallVector<mlir::Type> argTypes;
        for (auto paramItem : functionType->parameters)
        {
            auto type = getType(paramItem->type);
            if (paramItem->questionToken)
            {
                type = getOptionalType(type);
            }

            argTypes.push_back(type);
        }

        return mlir::FunctionType::get(builder.getContext(), argTypes, resultType);
    }

    mlir::Type getUnionType(UnionTypeNode unionTypeNode)
    {
        mlir::SmallVector<mlir::Type> types;
        auto oneType = true;
        mlir::Type currentType;
        for (auto typeItem : unionTypeNode->types)
        {
            auto type = getType(typeItem);
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            if (currentType && currentType != type)
            {
                oneType = false;
            }

            currentType = type;

            types.push_back(type);
        }

        if (oneType)
        {
            return currentType;
        }

        return getUnionType(types);
    }

    mlir_ts::UnionType getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::UnionType::get(builder.getContext(), types);
    }

    mlir_ts::IntersectionType getIntersectionType(IntersectionTypeNode intersectionTypeNode)
    {
        mlir::SmallVector<mlir::Type> types;
        for (auto typeItem : intersectionTypeNode->types)
        {
            auto type = getType(typeItem);
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            types.push_back(type);
        }

        return getIntersectionType(types);
    }

    mlir_ts::IntersectionType getIntersectionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::IntersectionType::get(builder.getContext(), types);
    }

    mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode)
    {
        return getType(parenthesizedTypeNode->type);
    }

    mlir::Type getLiteralType(LiteralTypeNode literalTypeNode)
    {
        GenContext genContext;
        genContext.dummyRun = true;
        genContext.allowPartialResolve = true;
        auto value = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
        auto type = value.getType();
        return type;
    }

    mlir_ts::OptionalType getOptionalType(mlir::Type type)
    {
        return mlir_ts::OptionalType::get(type);
    }

    mlir_ts::UndefPlaceHolderType getUndefPlaceHolderType()
    {
        return mlir_ts::UndefPlaceHolderType::get(builder.getContext());
    }

    mlir_ts::AnyType getAnyType()
    {
        return mlir_ts::AnyType::get(builder.getContext());
    }

    mlir::LogicalResult declare(VariableDeclarationDOM::TypePtr var, mlir::Value value, bool redeclare = false)
    {
        const auto &name = var->getName();
        /*
        if (!redeclare && symbolTable.count(name))
        {
            return mlir::failure();
        }
        */

        symbolTable.insert(name, {value, var});
        return mlir::success();
    }

    auto getNamespace() -> StringRef
    {
        if (currentNamespace->fullName.empty())
        {
            return "";
        }

        return currentNamespace->fullName;
    }

    auto getFullNamespaceName(StringRef name) -> StringRef
    {
        if (currentNamespace->fullName.empty())
        {
            return name;
        }

        std::string res;
        res += currentNamespace->fullName;
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

    auto getNamespaceByFullName(StringRef fullName) -> NamespaceInfo::TypePtr
    {
        return fullNamespacesMap.lookup(fullName);
    }

    auto getNamespaceMap() -> llvm::StringMap<NamespaceInfo::TypePtr> &
    {
        return currentNamespace->namespacesMap;
    }

    auto getFunctionMap() -> llvm::StringMap<mlir_ts::FuncOp> &
    {
        return currentNamespace->functionMap;
    }

    auto getGlobalsMap() -> llvm::StringMap<VariableDeclarationDOM::TypePtr> &
    {
        return currentNamespace->globalsMap;
    }

    auto getCaptureVarsMap() -> llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> &
    {
        return currentNamespace->captureVarsMap;
    }

    auto getClassesMap() -> llvm::StringMap<ClassInfo::TypePtr> &
    {
        return currentNamespace->classesMap;
    }

    auto getInterfacesMap() -> llvm::StringMap<InterfaceInfo::TypePtr> &
    {
        return currentNamespace->interfacesMap;
    }

    auto getEnumsMap() -> llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> &
    {
        return currentNamespace->enumsMap;
    }

    auto getTypeAliasMap() -> llvm::StringMap<mlir::Type> &
    {
        return currentNamespace->typeAliasMap;
    }

    auto getImportEqualsMap() -> llvm::StringMap<mlir::StringRef> &
    {
        return currentNamespace->importEqualsMap;
    }

    auto getClassByFullName(StringRef fullName) -> ClassInfo::TypePtr
    {
        return fullNameClassesMap.lookup(fullName);
    }

    auto getInterfaceByFullName(StringRef fullName) -> InterfaceInfo::TypePtr
    {
        return fullNameInterfacesMap.lookup(fullName);
    }

  protected:
    mlir::StringAttr getStringAttr(std::string text)
    {
        return builder.getStringAttr(text);
    }

    /// Helper conversion for a TypeScript AST location to an MLIR location.
    mlir::Location loc(TextRange loc)
    {
        // return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc->pos, loc->_end);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, loc->pos);
        return mlir::FileLineColLoc::get(builder.getContext(), builder.getIdentifier(fileName), posLineChar.line + 1,
                                         posLineChar.character + 1);
    }

    /// A "module" matches a TypeScript source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    mlir::StringRef fileName;

    /// An allocator used for alias names.
    llvm::BumpPtrAllocator stringAllocator;

    llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

    NamespaceInfo::TypePtr rootNamespace;

    NamespaceInfo::TypePtr currentNamespace;

    llvm::ScopedHashTable<StringRef, NamespaceInfo::TypePtr> fullNamespacesMap;

    llvm::ScopedHashTable<StringRef, ClassInfo::TypePtr> fullNameClassesMap;

    llvm::ScopedHashTable<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMap;

    llvm::ScopedHashTable<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMap;

    // helper to get line number
    Parser parser;
    ts::SourceFile sourceFile;

    mlir::OpBuilder::InsertPoint functionBeginPoint;

    std::string label;
};
} // namespace

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    auto showLineCharPos = false;

    Parser parser;
    auto sourceFile =
        parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);

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

            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S("(") << posLineChar.line + 1
              << S(":") << posLineChar.character + 1 << S(") - ") << child->_end << S("(") << endLineChar.line + 1 << S(":")
              << endLineChar.character << S(") ]") << std::endl;
        }
        else
        {
            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S(" - ") << child->_end << S(" ]")
              << std::endl;
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
    return wstos(s.str());
}

mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    Parser parser;
    auto sourceFile =
        parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);
    return MLIRGenImpl(context, fileName).mlirGenSourceFile(sourceFile);
}

} // namespace typescript
