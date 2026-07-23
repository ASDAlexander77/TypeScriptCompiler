#include "enums.h"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRGenStore.h"
#include "TypeScript/MLIRLogic/MLIRDeclarationPrinter.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"

#define DEBUG_TYPE "test"

namespace mlir_ts = mlir::typescript;
using namespace typescript;

// Characterization/regression tests for MLIRDeclarationPrinter - the component that
// re-serializes a compiled module's classes/interfaces/functions/types/enums/variables
// into TypeScript source text embedded in the binary's __decls global, so an importing
// module's `import '...'` can re-parse them. Every declaration kind it emits is marked
// `@dllimport` (see printBeforeDeclaration) - that marker, plus getting every
// layout-relevant modifier (abstract, static, optional, access level, extends/implements,
// index signatures) right, is what makes cross-module dynamic import actually work.
// A silent one-modifier omission here (e.g. the `abstract` bug fixed in PR #290) does not
// fail to compile - it makes the reimporting module reconstruct a DIFFERENT type, silently
// corrupting cross-module vtable layout / field offsets at runtime, which is exactly why a
// direct printer-output test (not just an end-to-end compile+run test) earns its keep here.
struct DeclarationPrinterTest : public testing::Test
{
public:
    void SetUp() override
    {
        context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    }

    mlir::MLIRContext *getContext()
    {
        return &context;
    }

    // ---- generic printer-invocation helpers ----

    template <typename Fn> std::string printed(Fn fn)
    {
        std::string result;
        llvm::raw_string_ostream rso(result);
        MLIRDeclarationPrinter dp(rso);
        fn(dp);
        return rso.str();
    }

    // ---- namespace helpers ----

    NamespaceInfo::TypePtr makeNamespace(StringRef fullName)
    {
        auto ns = std::make_shared<NamespaceInfo>();
        ns->name = fullName;
        ns->fullName = fullName;
        return ns;
    }

    NamespaceInfo::TypePtr noNamespace()
    {
        return NamespaceInfo::TypePtr();
    }

    // ---- type helpers (mirrors unittests/MLIRGen/TypeToString.cpp) ----

    template <typename T> T get()
    {
        return T::get(getContext());
    }

    template <typename T, typename E> T getE()
    {
        return T::get(getContext(), get<E>());
    }

    mlir_ts::FunctionType getF(::llvm::ArrayRef<mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> results, bool isVarArg = false)
    {
        return mlir_ts::FunctionType::get(getContext(), inputs, results, isVarArg);
    }

    mlir_ts::OptionalType getOpt(mlir::Type element)
    {
        return mlir_ts::OptionalType::get(getContext(), element);
    }

    mlir::FlatSymbolRefAttr sym(StringRef name)
    {
        return mlir::FlatSymbolRefAttr::get(getContext(), name);
    }

    mlir::StringAttr strAttr(StringRef name)
    {
        return mlir::StringAttr::get(getContext(), name);
    }

    // ---- class helpers ----

    ClassInfo::TypePtr makeClass(StringRef fullName, NamespaceInfo::TypePtr ns = NamespaceInfo::TypePtr())
    {
        auto ci = std::make_shared<ClassInfo>();
        ci->name = fullName;
        ci->fullName = fullName;
        ci->elementNamespace = ns;
        auto nameAttr = sym(fullName);
        auto storageType = mlir_ts::ClassStorageType::get(getContext(), nameAttr);
        ci->classType = mlir_ts::ClassType::get(getContext(), nameAttr, storageType);
        return ci;
    }

    void setClassFields(ClassInfo::TypePtr ci, ::llvm::ArrayRef<mlir_ts::FieldInfo> fields)
    {
        mlir::cast<mlir_ts::ClassStorageType>(ci->classType.getStorageType()).setFields(fields);
    }

    mlir_ts::FieldInfo classField(mlir::Attribute id, mlir::Type type, bool isConditional = false,
                                   mlir_ts::AccessLevel access = mlir_ts::AccessLevel::Public)
    {
        return mlir_ts::FieldInfo(id, type, isConditional, access);
    }

    MethodInfo method(StringRef name, mlir_ts::FunctionType funcType, bool isStatic = false, bool isAbstract = false,
                       mlir_ts::AccessLevel access = mlir_ts::AccessLevel::Public)
    {
        MethodInfo m;
        m.name = name.str();
        m.funcType = funcType;
        m.funcName = name.str();
        m.isStatic = isStatic;
        m.isVirtual = true;
        m.isAbstract = isAbstract;
        m.virtualIndex = 0;
        m.orderWeight = 0;
        m.accessLevel = access;
        return m;
    }

    FunctionEntry entry(StringRef name, mlir_ts::FunctionType funcType)
    {
        FunctionEntry e;
        e.name = name.str();
        e.funcType = funcType;
        return e;
    }

    FunctionEntry noEntry()
    {
        return FunctionEntry{};
    }

    AccessorInfo accessor(StringRef name, FunctionEntry get, FunctionEntry set, bool isStatic = false,
                           mlir_ts::AccessLevel getAccess = mlir_ts::AccessLevel::Public,
                           mlir_ts::AccessLevel setAccess = mlir_ts::AccessLevel::Public)
    {
        AccessorInfo a;
        a.name = name.str();
        a.get = get;
        a.set = set;
        a.isStatic = isStatic;
        a.isVirtual = true;
        a.isAbstract = false;
        a.getAccessLevel = getAccess;
        a.setAccessLevel = setAccess;
        return a;
    }

    StaticFieldInfo staticField(mlir::Attribute id, mlir::Type type, mlir_ts::AccessLevel access = mlir_ts::AccessLevel::Public)
    {
        StaticFieldInfo sf;
        sf.id = id;
        sf.type = type;
        sf.globalVariableName = "";
        sf.virtualIndex = 0;
        sf.accessLevel = access;
        return sf;
    }

    IndexInfo classIndex(mlir_ts::FunctionType sig)
    {
        IndexInfo ii;
        ii.indexSignature = sig;
        ii.getAccessLevel = mlir_ts::AccessLevel::Public;
        ii.setAccessLevel = mlir_ts::AccessLevel::Public;
        return ii;
    }

    ImplementInfo implement(InterfaceInfo::TypePtr iface)
    {
        ImplementInfo im;
        im.interface = iface;
        im.virtualIndex = 0;
        im.processed = true;
        return im;
    }

    std::string printClass(ClassInfo::TypePtr ci)
    {
        return printed([&](MLIRDeclarationPrinter &dp) { dp.print(ci); });
    }

    // ---- interface helpers ----

    InterfaceInfo::TypePtr makeInterface(StringRef fullName, NamespaceInfo::TypePtr ns = NamespaceInfo::TypePtr())
    {
        auto ii = std::make_shared<InterfaceInfo>();
        ii->name = fullName;
        ii->fullName = fullName;
        ii->elementNamespace = ns;
        ii->interfaceType = mlir_ts::InterfaceType::get(getContext(), sym(fullName));
        return ii;
    }

    InterfaceFieldInfo ifield(mlir::Attribute id, mlir::Type type, bool isConditional = false)
    {
        InterfaceFieldInfo f;
        f.id = id;
        f.type = type;
        f.isConditional = isConditional;
        f.interfacePosIndex = 0;
        f.virtualIndex = 0;
        return f;
    }

    InterfaceMethodInfo imethod(StringRef name, mlir_ts::FunctionType funcType, bool isConditional = false)
    {
        InterfaceMethodInfo m;
        m.name = name.str();
        m.funcType = funcType;
        m.isConditional = isConditional;
        m.interfacePosIndex = 0;
        m.virtualIndex = 0;
        return m;
    }

    InterfaceIndexInfo interfaceIndex(mlir_ts::FunctionType sig)
    {
        InterfaceIndexInfo ii;
        ii.indexSignature = sig;
        ii.getMethod = "";
        ii.setMethod = "";
        return ii;
    }

    InterfaceInfo::InterfaceInfoWithOffset extend(int offset, InterfaceInfo::TypePtr iface)
    {
        return {offset, iface};
    }

    std::string printInterface(InterfaceInfo::TypePtr ii)
    {
        return printed([&](MLIRDeclarationPrinter &dp) { dp.print(ii); });
    }

    mlir::MLIRContext context;
};

// ---------------------------------------------------------------------------
// @dllimport contract: every declaration kind the printer emits is meant to be
// re-parsed by an importing module's `import '...'`, and printBeforeDeclaration
// is the ONLY thing that marks the text that way.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, type_declaration_has_dllimport_marker)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printTypeDeclaration("Foo", noNamespace(), get<mlir_ts::NumberType>());
    });
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

TEST_F(DeclarationPrinterTest, enum_declaration_has_dllimport_marker)
{
    auto dict = mlir::DictionaryAttr::get(getContext(), {mlir::NamedAttribute(strAttr("A"), mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 1))});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printEnum("E", noNamespace(), dict); });
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

TEST_F(DeclarationPrinterTest, variable_declaration_has_dllimport_marker)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printVariableDeclaration("x", noNamespace(), get<mlir_ts::NumberType>(), /*isConst*/ true);
    });
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

TEST_F(DeclarationPrinterTest, function_declaration_has_dllimport_marker)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.print("foo", noNamespace(), getF({}, {}));
    });
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

TEST_F(DeclarationPrinterTest, class_declaration_has_dllimport_marker)
{
    auto text = printClass(makeClass("Foo"));
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

TEST_F(DeclarationPrinterTest, interface_declaration_has_dllimport_marker)
{
    auto text = printInterface(makeInterface("Foo"));
    EXPECT_THAT(text, testing::HasSubstr("@dllimport"));
}

// printGenericClass deliberately omits @dllimport: a generic class has no compiled
// body for any instantiation to import against, so the importer re-instantiates from
// raw source text instead - see the comment on printGenericClass itself. Locking this
// down as a test, not just a comment, so a future "make it consistent with the other
// print* methods" cleanup doesn't silently break generic cross-module imports.
TEST_F(DeclarationPrinterTest, generic_class_declaration_has_no_dllimport_marker)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printGenericClass(noNamespace(), "class Box<T> { value: T; }");
    });
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("@dllimport")));
    EXPECT_THAT(text, testing::HasSubstr("class Box<T> { value: T; }"));
}

// ---------------------------------------------------------------------------
// Namespace wrapping: present when there's a real namespace, absent otherwise -
// exercised once per print* entry point that takes a NamespaceInfo, since each
// calls printNamespaceBegin/End independently.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, class_wraps_in_namespace_when_present)
{
    auto text = printClass(makeClass("M.Foo", makeNamespace("M")));
    EXPECT_THAT(text, testing::HasSubstr("namespace M {"));
}

TEST_F(DeclarationPrinterTest, class_has_no_namespace_wrapper_when_absent)
{
    auto text = printClass(makeClass("Foo", noNamespace()));
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("namespace")));
}

TEST_F(DeclarationPrinterTest, interface_wraps_in_namespace_when_present)
{
    auto text = printInterface(makeInterface("M.Foo", makeNamespace("M")));
    EXPECT_THAT(text, testing::HasSubstr("namespace M {"));
}

// ---------------------------------------------------------------------------
// Class printing: this is the arc PR #290 fixed (abstract modifier dropped),
// so abstract coverage on both the class and the method comes first.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, abstract_class_prints_abstract_keyword)
{
    auto ci = makeClass("Shape");
    ci->isAbstract = true;
    setClassFields(ci, {});
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("abstract class Shape"));
}

TEST_F(DeclarationPrinterTest, concrete_class_has_no_abstract_keyword)
{
    auto ci = makeClass("Shape");
    ci->isAbstract = false;
    setClassFields(ci, {});
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("abstract")));
    EXPECT_THAT(text, testing::HasSubstr("class Shape"));
}

TEST_F(DeclarationPrinterTest, abstract_method_prints_abstract_keyword)
{
    auto ci = makeClass("Shape");
    setClassFields(ci, {});
    // sole param's type equals the class's own type -> elided as the implicit `this`
    // (see class_method_omits_this_parameter below - printParams silently drops it,
    // it does not print the literal word "this")
    ci->methods.push_back(method("area", getF({ci->classType}, {get<mlir_ts::NumberType>()}), false, /*isAbstract*/ true));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("abstract area() : number"));
}

TEST_F(DeclarationPrinterTest, concrete_method_has_no_abstract_keyword)
{
    auto ci = makeClass("Shape");
    setClassFields(ci, {});
    ci->methods.push_back(method("area", getF({ci->classType}, {get<mlir_ts::NumberType>()}), false, /*isAbstract*/ false));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("abstract")));
    EXPECT_THAT(text, testing::HasSubstr("area() : number"));
}

TEST_F(DeclarationPrinterTest, class_prints_single_extends)
{
    auto base = makeClass("Base");
    setClassFields(base, {});
    auto derived = makeClass("Derived");
    derived->baseClasses.push_back(base);
    setClassFields(derived, {});
    auto text = printClass(derived);
    EXPECT_THAT(text, testing::HasSubstr("class Derived extends Base"));
}

TEST_F(DeclarationPrinterTest, class_prints_multiple_extends_comma_separated)
{
    auto base1 = makeClass("Base1");
    setClassFields(base1, {});
    auto base2 = makeClass("Base2");
    setClassFields(base2, {});
    auto derived = makeClass("Derived");
    derived->baseClasses.push_back(base1);
    derived->baseClasses.push_back(base2);
    setClassFields(derived, {});
    auto text = printClass(derived);
    EXPECT_THAT(text, testing::HasSubstr("extends Base1, Base2"));
}

TEST_F(DeclarationPrinterTest, class_prints_single_implements)
{
    auto iface = makeInterface("Describable");
    auto ci = makeClass("Shape");
    ci->implements.push_back(implement(iface));
    setClassFields(ci, {});
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("class Shape implements Describable"));
}

TEST_F(DeclarationPrinterTest, class_prints_multiple_implements_comma_separated)
{
    auto iface1 = makeInterface("Describable");
    auto iface2 = makeInterface("Sizeable");
    auto ci = makeClass("Shape");
    ci->implements.push_back(implement(iface1));
    ci->implements.push_back(implement(iface2));
    setClassFields(ci, {});
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("implements Describable, Sizeable"));
}

TEST_F(DeclarationPrinterTest, class_prints_static_field_with_access_level)
{
    auto ci = makeClass("Counter");
    setClassFields(ci, {});
    ci->staticFields.push_back(staticField(strAttr("total"), get<mlir_ts::NumberType>(), mlir_ts::AccessLevel::Protected));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("protected static total: number;"));
}

TEST_F(DeclarationPrinterTest, class_prints_instance_field_with_access_level_and_optional)
{
    auto ci = makeClass("Point");
    setClassFields(ci, {classField(strAttr("x"), get<mlir_ts::NumberType>(), /*isConditional*/ true, mlir_ts::AccessLevel::Private)});
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("private x?: number;"));
}

// storageType.getFields() embeds each base class's storage as a synthetic first field
// whose id equals the base's fullName (mlirGenClassHeritageClause) - printing it back
// as a real field would shift every subsequent field's offset for the importer. See
// the comment on this exact check in DeclarationPrinter.cpp::print(ClassInfo::TypePtr).
TEST_F(DeclarationPrinterTest, class_filters_out_synthetic_base_class_storage_field)
{
    auto base = makeClass("Base");
    setClassFields(base, {});
    auto derived = makeClass("Derived");
    derived->baseClasses.push_back(base);
    setClassFields(derived, {
        classField(strAttr("Base"), base->classType, false, mlir_ts::AccessLevel::Public),
        classField(strAttr("side"), get<mlir_ts::NumberType>(), false, mlir_ts::AccessLevel::Public),
    });
    auto text = printClass(derived);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("Base:")));
    EXPECT_THAT(text, testing::HasSubstr("side: number;"));
}

TEST_F(DeclarationPrinterTest, class_filters_out_dot_prefixed_synthetic_fields)
{
    auto ci = makeClass("Foo");
    setClassFields(ci, {
        classField(strAttr(".vtbl"), get<mlir_ts::AnyType>()),
        classField(strAttr("real"), get<mlir_ts::NumberType>()),
    });
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr(".vtbl")));
    EXPECT_THAT(text, testing::HasSubstr("real: number;"));
}

TEST_F(DeclarationPrinterTest, class_filters_out_dot_prefixed_synthetic_methods)
{
    auto ci = makeClass("Foo");
    setClassFields(ci, {});
    ci->methods.push_back(method(".instanceOf", getF({ci->classType, get<mlir_ts::StringType>()}, {get<mlir_ts::BooleanType>()})));
    ci->methods.push_back(method("real", getF({ci->classType}, {})));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr(".instanceOf")));
    EXPECT_THAT(text, testing::HasSubstr("real()"));
}

// index signature ([x: T]: U;) needs its own declaration text - unlike accessors, its
// get/set methods are ordinary named methods already covered by the methods loop, but
// without the signature itself a reimporting module's ClassInfo::indexes stays empty
// and obj[i]/super[i] can't resolve (see cross-module-class-indexer-shared-gap-fix).
TEST_F(DeclarationPrinterTest, class_prints_index_signature)
{
    auto ci = makeClass("Container");
    setClassFields(ci, {});
    ci->indexes.push_back(classIndex(getF({get<mlir_ts::NumberType>()}, {get<mlir_ts::StringType>()})));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("[p0: number]: string;"));
}

TEST_F(DeclarationPrinterTest, class_skips_empty_index_signature)
{
    auto ci = makeClass("Container");
    setClassFields(ci, {});
    ci->indexes.push_back(classIndex(mlir_ts::FunctionType()));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("p0")));
}

TEST_F(DeclarationPrinterTest, class_prints_static_method)
{
    auto ci = makeClass("Foo");
    setClassFields(ci, {});
    // static methods have no implicit `this` param
    ci->methods.push_back(method("create", getF({}, {get<mlir_ts::NumberType>()}), /*isStatic*/ true));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("static create() : number"));
}

// get/set accessors are ALSO registered in classType->methods under their mangled
// "get_x"/"set_x" funcOp name (for same-file/JIT lookup) - the class printer must
// exclude those from the plain methods loop and print them via classType->accessors
// instead using real get/set syntax, or a reimporting module never populates its
// accessors list and property-style access (obj.x) fails to resolve.
TEST_F(DeclarationPrinterTest, class_excludes_accessor_backing_methods_from_methods_list)
{
    auto ci = makeClass("Thermometer");
    setClassFields(ci, {});
    auto getFn = getF({ci->classType}, {get<mlir_ts::NumberType>()});
    ci->methods.push_back(method("get_celsius", getFn));
    ci->accessors.push_back(accessor("celsius", entry("get_celsius", getFn), noEntry()));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("get_celsius")));
    EXPECT_THAT(text, testing::HasSubstr("get celsius()"));
}

TEST_F(DeclarationPrinterTest, class_prints_get_only_accessor)
{
    auto ci = makeClass("Thermometer");
    setClassFields(ci, {});
    auto getFn = getF({ci->classType}, {get<mlir_ts::NumberType>()});
    ci->accessors.push_back(accessor("celsius", entry("get_celsius", getFn), noEntry(), false, mlir_ts::AccessLevel::Protected));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("protected get celsius() : number"));
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("set celsius")));
}

TEST_F(DeclarationPrinterTest, class_prints_set_only_accessor)
{
    auto ci = makeClass("Thermometer");
    setClassFields(ci, {});
    auto setFn = getF({ci->classType, get<mlir_ts::NumberType>()}, {});
    ci->accessors.push_back(accessor("celsius", noEntry(), entry("set_celsius", setFn), false, mlir_ts::AccessLevel::Public, mlir_ts::AccessLevel::Private));
    auto text = printClass(ci);
    // param 0 (`this`) is elided, so the value param keeps its original index p1
    EXPECT_THAT(text, testing::HasSubstr("private set celsius(p1 : number)"));
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("get celsius")));
}

TEST_F(DeclarationPrinterTest, class_prints_get_and_set_accessor_pair)
{
    auto ci = makeClass("Thermometer");
    setClassFields(ci, {});
    auto getFn = getF({ci->classType}, {get<mlir_ts::NumberType>()});
    auto setFn = getF({ci->classType, get<mlir_ts::NumberType>()}, {});
    ci->accessors.push_back(accessor("celsius", entry("get_celsius", getFn), entry("set_celsius", setFn)));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("get celsius() : number"));
    EXPECT_THAT(text, testing::HasSubstr("set celsius(p1 : number)"));
}

TEST_F(DeclarationPrinterTest, class_method_omits_this_parameter)
{
    auto ci = makeClass("Foo");
    setClassFields(ci, {});
    // first param type equals the class's own type -> elided as the implicit `this`
    ci->methods.push_back(method("bar", getF({ci->classType, get<mlir_ts::NumberType>()}, {})));
    auto text = printClass(ci);
    EXPECT_THAT(text, testing::HasSubstr("bar(p1 : number)"));
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("p0")));
}

// ---------------------------------------------------------------------------
// Interface printing
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, interface_prints_field_with_optional)
{
    auto ii = makeInterface("Point");
    ii->fields.push_back(ifield(strAttr("x"), get<mlir_ts::NumberType>(), /*isConditional*/ true));
    auto text = printInterface(ii);
    EXPECT_THAT(text, testing::HasSubstr("x?: number;"));
}

TEST_F(DeclarationPrinterTest, interface_filters_dot_prefixed_field)
{
    auto ii = makeInterface("Point");
    ii->fields.push_back(ifield(strAttr(".hidden"), get<mlir_ts::NumberType>()));
    ii->fields.push_back(ifield(strAttr("x"), get<mlir_ts::NumberType>()));
    auto text = printInterface(ii);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr(".hidden")));
    EXPECT_THAT(text, testing::HasSubstr("x: number;"));
}

TEST_F(DeclarationPrinterTest, interface_prints_single_extends)
{
    auto base = makeInterface("Base");
    auto derived = makeInterface("Derived");
    derived->extends.push_back(extend(0, base));
    auto text = printInterface(derived);
    EXPECT_THAT(text, testing::HasSubstr("interface Derived extends Base"));
}

TEST_F(DeclarationPrinterTest, interface_prints_multiple_extends_comma_separated)
{
    auto base1 = makeInterface("Base1");
    auto base2 = makeInterface("Base2");
    auto derived = makeInterface("Derived");
    derived->extends.push_back(extend(0, base1));
    derived->extends.push_back(extend(0, base2));
    auto text = printInterface(derived);
    EXPECT_THAT(text, testing::HasSubstr("extends Base1, Base2"));
}

TEST_F(DeclarationPrinterTest, interface_prints_method)
{
    auto ii = makeInterface("Describable");
    ii->methods.push_back(imethod("describe", getF({}, {get<mlir_ts::StringType>()})));
    auto text = printInterface(ii);
    // unlike fields/indexers, printMethod's caller does not append a trailing ";"
    EXPECT_THAT(text, testing::HasSubstr("describe() : string"));
}

TEST_F(DeclarationPrinterTest, interface_filters_dot_prefixed_method)
{
    auto ii = makeInterface("Describable");
    ii->methods.push_back(imethod(".internal", getF({}, {})));
    ii->methods.push_back(imethod("describe", getF({}, {get<mlir_ts::StringType>()})));
    auto text = printInterface(ii);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr(".internal")));
    EXPECT_THAT(text, testing::HasSubstr("describe()"));
}

// Interface index signatures are built with an opaque `this` prepended as input(0)
// (getInterfaceMethodNameAndType with thisType set) - so the real index-argument type
// is input(1), unlike a class's plain (arg)->result signature. Getting this offset
// wrong silently prints the WRONG argument type for obj[i] through an interface.
TEST_F(DeclarationPrinterTest, interface_prints_index_signature_skipping_opaque_this)
{
    auto ii = makeInterface("Container");
    auto opaque = get<mlir_ts::OpaqueType>();
    ii->indexes.push_back(interfaceIndex(getF({opaque, get<mlir_ts::NumberType>()}, {get<mlir_ts::StringType>()})));
    auto text = printInterface(ii);
    EXPECT_THAT(text, testing::HasSubstr("[p0: number]: string;"));
}

TEST_F(DeclarationPrinterTest, interface_skips_empty_index_signature)
{
    auto ii = makeInterface("Container");
    ii->indexes.push_back(interfaceIndex(mlir_ts::FunctionType()));
    auto text = printInterface(ii);
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("p0")));
}

// ---------------------------------------------------------------------------
// Enum printing: member-name text plus each supported value-attribute kind.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, enum_prints_integer_value)
{
    auto dict = mlir::DictionaryAttr::get(getContext(),
        {mlir::NamedAttribute(strAttr("Red"), mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 1))});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printEnum("Color", noNamespace(), dict); });
    EXPECT_THAT(text, testing::HasSubstr("enum Color"));
    EXPECT_THAT(text, testing::HasSubstr("Red = 1,"));
}

TEST_F(DeclarationPrinterTest, enum_prints_boolean_value_without_type_suffix)
{
    auto dict = mlir::DictionaryAttr::get(getContext(),
        {mlir::NamedAttribute(strAttr("Flag"), mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), 1))});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printEnum("E", noNamespace(), dict); });
    EXPECT_THAT(text, testing::HasSubstr("Flag = true,"));
}

TEST_F(DeclarationPrinterTest, enum_prints_string_value)
{
    auto dict = mlir::DictionaryAttr::get(getContext(),
        {mlir::NamedAttribute(strAttr("Name"), strAttr("hello"))});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printEnum("E", noNamespace(), dict); });
    EXPECT_THAT(text, testing::HasSubstr("Name = \"hello\","));
}

// ---------------------------------------------------------------------------
// Variable printing: const vs let, and the @boxed marker an inferred (untyped)
// object-literal export needs so the importer's isDynamicImport load knows to
// dereference one extra level instead of reading the tuple inline (see the
// comment on printVariableDeclaration itself).
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, variable_prints_const_keyword)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printVariableDeclaration("x", noNamespace(), get<mlir_ts::NumberType>(), /*isConst*/ true);
    });
    EXPECT_THAT(text, testing::HasSubstr("const x : number;"));
}

TEST_F(DeclarationPrinterTest, variable_prints_let_keyword)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printVariableDeclaration("x", noNamespace(), get<mlir_ts::NumberType>(), /*isConst*/ false);
    });
    EXPECT_THAT(text, testing::HasSubstr("let x : number;"));
}

TEST_F(DeclarationPrinterTest, variable_of_plain_type_has_no_boxed_marker)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printVariableDeclaration("x", noNamespace(), get<mlir_ts::NumberType>(), true);
    });
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("@boxed")));
}

TEST_F(DeclarationPrinterTest, variable_of_object_with_tuple_storage_has_boxed_marker)
{
    llvm::SmallVector<mlir_ts::FieldInfo> fields{classField(strAttr("x"), get<mlir_ts::NumberType>())};
    auto tuple = mlir_ts::TupleType::get(getContext(), fields);
    auto objType = getE<mlir_ts::ObjectType, mlir_ts::AnyType>();
    // ObjectType wraps AnyType by default via getE<>; rebuild it explicitly over the tuple storage.
    objType = mlir_ts::ObjectType::get(getContext(), tuple);
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printVariableDeclaration("obj", noNamespace(), objType, true); });
    EXPECT_THAT(text, testing::HasSubstr("@boxed"));
}

TEST_F(DeclarationPrinterTest, variable_of_object_with_non_tuple_storage_has_no_boxed_marker)
{
    // ObjectType over a plain (non-tuple/const-tuple/object-storage) element type - the
    // @boxed check is deliberately narrow to those three storage kinds.
    auto objType = getE<mlir_ts::ObjectType, mlir_ts::AnyType>();
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.printVariableDeclaration("obj", noNamespace(), objType, true); });
    EXPECT_THAT(text, testing::Not(testing::HasSubstr("@boxed")));
}

// ---------------------------------------------------------------------------
// Function printing: params (with optional-param and this-elision handling
// shared with printMethod via printParams), and return-type presence.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, function_prints_params_and_return_type)
{
    auto funcType = getF({get<mlir_ts::NumberType>(), get<mlir_ts::StringType>()}, {get<mlir_ts::BooleanType>()});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.print("foo", noNamespace(), funcType); });
    EXPECT_THAT(text, testing::HasSubstr("function foo(p0 : number, p1 : string) : boolean;"));
}

TEST_F(DeclarationPrinterTest, function_with_no_return_type_omits_colon)
{
    auto funcType = getF({}, {});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.print("foo", noNamespace(), funcType); });
    EXPECT_THAT(text, testing::HasSubstr("function foo();"));
    EXPECT_THAT(text, testing::Not(testing::HasSubstr(" : ")));
}

TEST_F(DeclarationPrinterTest, function_prints_optional_param_with_question_mark)
{
    auto funcType = getF({getOpt(get<mlir_ts::NumberType>())}, {});
    auto text = printed([&](MLIRDeclarationPrinter &dp) { dp.print("foo", noNamespace(), funcType); });
    EXPECT_THAT(text, testing::HasSubstr("foo(p0? : number)"));
}

// ---------------------------------------------------------------------------
// printGenericClass: raw source-text passthrough (see the @dllimport test above
// for why it deliberately skips printBeforeDeclaration), still namespace-wrapped.
// ---------------------------------------------------------------------------

TEST_F(DeclarationPrinterTest, generic_class_wraps_raw_text_in_namespace)
{
    auto text = printed([&](MLIRDeclarationPrinter &dp) {
        dp.printGenericClass(makeNamespace("M"), "class Box<T> { value: T; }");
    });
    EXPECT_THAT(text, testing::HasSubstr("namespace M {"));
    EXPECT_THAT(text, testing::HasSubstr("class Box<T> { value: T; }"));
}
