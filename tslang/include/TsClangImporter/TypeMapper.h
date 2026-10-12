#ifndef TSCLANGIMPORTER_TYPEMAPPER_H
#define TSCLANGIMPORTER_TYPEMAPPER_H

#include "TsClangImporter/BindingModel.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

namespace clang
{
class Sema;
}
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"

#include <utility>
#include <vector>

namespace tsbindgen
{

// C type -> tslang type, for one translation unit. Widths come from the ASTContext, so they are the
// target's: C `long` is i32 for x86_64-pc-windows-msvc and i64 for x86_64-linux-gnu.
class TypeMapper
{
  public:
    enum class Use
    {
        Parameter,
        Result,
        Field,
        Alias // the right-hand side of a typedef
    };

    explicit TypeMapper(clang::ASTContext &context) : context(context)
    {
    }

    // A pointer parameter or field is nullable (`T | null`); a pointer result isn't.
    TsType map(clang::QualType type, Use use);

    // Why a complete struct cannot be a named tuple ("" if it can). Cached; a struct that refers to
    // itself through a pointer counts as mappable while it is being checked.
    std::string recordSkipReason(const clang::RecordDecl *record);

    // A complete struct's fields. A pointer field that leads back to this struct is Opaque:
    // tslang cannot declare a type that refers to itself.
    std::vector<Field> mapFields(const clang::RecordDecl *definition);

    // The struct/enum's own name, or the typedef that names it when it has none ("" for neither).
    static std::string tagName(const clang::TagDecl *tag);
    // tagName with its namespaces and classes for a C++ struct out of the file scope (`nvrhi::Color`)
    static std::string qualifiedTagName(const clang::TagDecl *tag);
    static std::string recordKey(const clang::RecordDecl *record);
    static std::string enumKey(const clang::EnumDecl *enumDecl);
    static std::string typedefKey(const clang::TypedefNameDecl *typedefDecl);

    // `typedef struct S S;` and `typedef struct { ... } S;` name the struct itself: no alias is emitted.
    static bool namesItsTag(const clang::TypedefNameDecl *typedefDecl);

    // A C enum is a TS enum only when it is int-sized and not given a fixed underlying type other
    // than int; anything else is a type alias plus constants.
    bool isPlainEnum(const clang::EnumDecl *enumDecl);

    // Whether a struct can have a TS declaration of its own: in C, any; in C++, one in a namespace or
    // a class too (declared under its own name), but not a template, a template specialization
    // (std::cmatch) or a class inside one, nor a class local to a function.
    bool isDeclarable(const clang::RecordDecl *record);

    // Why a C++ class can't be a @cpp class ("" if it can): it has to be polymorphic, not a
    // template, on a 64-bit target, with at most one base, itself bindable and not virtual - the
    // layout a @cpp class has (the vtable pointer first, a base object the prefix of a derived one).
    std::string classSkipReason(const clang::CXXRecordDecl *record);

    // "class:nvrhi::IDevice"
    static std::string classKey(const clang::CXXRecordDecl *record);

    // A C++ record returned by value, as a method declared to return a tuple returns it: the
    // record's fields, laid out as a tuple's, a pointer as `Opaque | null`. `nonTrivial` is set
    // when C++ can't pass it in registers (a destructor or a copy constructor: @cppnontrivial).
    TsType mapRecordResult(clang::QualType type, bool &nonTrivial);

    // clang's semantic analysis, to complete a class template specialization only ever named as a
    // result (`RefCountPtr<IBuffer> createBuffer(...)` doesn't instantiate it)
    void setSema(clang::Sema *value)
    {
        sema = value;
    }

    // Every struct and enum a mapped type has named, by key. The AST traversal does not reach a
    // struct first named inside a prototype (`void f(struct S *s);`), so the parser adds these.
    const std::vector<std::pair<std::string, const clang::TagDecl *>> &referencedTags() const
    {
        return referenced;
    }

  private:
    TsType mapType(clang::QualType type, Use use);
    std::string referTo(const clang::TagDecl *tag);
    TsType mapPointer(clang::QualType pointee, Use use);
    TsType mapBuiltin(clang::QualType canonical, Use use);
    std::string signatureComment(const clang::FunctionProtoType *function);
    bool isSystem(const clang::Decl *decl);
    const clang::RecordDecl *recordBehind(clang::QualType type);
    bool leadsToOwner(clang::QualType type);

    clang::ASTContext &context;
    clang::Sema *sema = nullptr;
    llvm::DenseMap<const clang::RecordDecl *, std::string> recordReasons;
    llvm::DenseMap<const clang::CXXRecordDecl *, std::string> classReasons;
    std::vector<std::pair<std::string, const clang::TagDecl *>> referenced;
    llvm::StringSet<> referencedKeys;
    std::vector<const clang::RecordDecl *> owners; // structs whose fields are being mapped
};

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_TYPEMAPPER_H
