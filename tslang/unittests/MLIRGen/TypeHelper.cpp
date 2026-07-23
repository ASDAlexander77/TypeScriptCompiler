#include "enums.h"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"

#include "mlir/IR/MLIRContext.h"

#include "gmock/gmock.h"

#define DEBUG_TYPE "test"

namespace mlir_ts = mlir::typescript;
using namespace typescript;

// Direct unit coverage for the pure type-algebra helpers in MLIRTypeHelper.h and
// MLIRCodeLogic.h - previously zero direct coverage despite being the core
// assignability/widening/merging logic every cast, union, and field-access decision
// in the compiler routes through. canWideTypeWithoutDataLoss in particular is the
// function underlying the still-open interface-field-width-coercion bug (an
// int-literal field cast to an interface's `number` reads garbage after mutation) -
// these tests characterize its CURRENT behavior precisely so a future fix has a
// baseline to diff against, and so a future unrelated change can't silently alter
// widening rules without a test noticing.
struct MLIRTypeHelperTest : public testing::Test
{
public:
    void SetUp() override
    {
        context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
        compileOptions.sizeBits = 64;
    }

    mlir::MLIRContext *getContext()
    {
        return &context;
    }

    MLIRTypeHelper mth()
    {
        return MLIRTypeHelper(getContext(), compileOptions);
    }

    mlir::Location loc()
    {
        return mlir::UnknownLoc::get(getContext());
    }

    template <typename T> T get()
    {
        return T::get(getContext());
    }

    mlir::IntegerType si(unsigned width)
    {
        return mlir::IntegerType::get(getContext(), width, mlir::IntegerType::Signed);
    }

    mlir::IntegerType ui(unsigned width)
    {
        return mlir::IntegerType::get(getContext(), width, mlir::IntegerType::Unsigned);
    }

    mlir::IntegerType signless(unsigned width)
    {
        return mlir::IntegerType::get(getContext(), width, mlir::IntegerType::Signless);
    }

    mlir::FloatType f32()
    {
        return mlir::Float32Type::get(getContext());
    }

    mlir::FloatType f64()
    {
        return mlir::Float64Type::get(getContext());
    }

    mlir_ts::LiteralType literalInt(int64_t value, mlir::Type elementType)
    {
        return mlir_ts::LiteralType::get(mlir::IntegerAttr::get(elementType, value), elementType);
    }

    mlir::FlatSymbolRefAttr sym(StringRef name)
    {
        return mlir::FlatSymbolRefAttr::get(getContext(), name);
    }

    mlir::StringAttr strAttr(StringRef name)
    {
        return mlir::StringAttr::get(getContext(), name);
    }

    mlir_ts::FieldInfo field(mlir::Attribute id, mlir::Type type, bool isConditional = false,
                              mlir_ts::AccessLevel access = mlir_ts::AccessLevel::Public)
    {
        return mlir_ts::FieldInfo(id, type, isConditional, access);
    }

    mlir_ts::TupleType tuple(::llvm::ArrayRef<mlir_ts::FieldInfo> fields)
    {
        return mlir_ts::TupleType::get(getContext(), fields);
    }

    mlir::MLIRContext context;
    CompileOptions compileOptions{};
};

// ---------------------------------------------------------------------------
// canWideTypeWithoutDataLoss - the core widening-compatibility check.
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, widen_null_types_reject)
{
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(mlir::Type(), get<mlir_ts::NumberType>()));
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::NumberType>(), mlir::Type()));
}

TEST_F(MLIRTypeHelperTest, widen_identical_type_accepts)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(32), si(32)));
}

TEST_F(MLIRTypeHelperTest, widen_integer_to_number_accepts)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(32), get<mlir_ts::NumberType>()));
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(get<mlir::IndexType>(), get<mlir_ts::NumberType>()));
}

TEST_F(MLIRTypeHelperTest, widen_narrower_same_signed_int_accepts)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(8), si(32)));
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(ui(8), ui(32)));
}

TEST_F(MLIRTypeHelperTest, widen_wider_to_narrower_same_signed_int_rejects)
{
    // narrowing always loses data, regardless of signedness match
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(si(32), si(8)));
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(ui(32), ui(8)));
}

TEST_F(MLIRTypeHelperTest, widen_same_width_different_signedness_rejects)
{
    // si32 -> ui32: same width, no bit-count gap to hide the sign reinterpretation
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(si(32), ui(32)));
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(ui(32), si(32)));
}

TEST_F(MLIRTypeHelperTest, widen_signless_operand_accepts_regardless_of_dest_signedness)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(signless(8), si(32)));
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(signless(8), ui(32)));
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(8), signless(32)));
}

TEST_F(MLIRTypeHelperTest, widen_different_signedness_accepts_if_gap_exceeds_one_bit)
{
    // si8 -> ui32: different signedness, but the >1-bit gap rule (line ~1913) allows it
    // anyway - documenting this as CURRENT behavior, not necessarily obviously correct.
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(8), ui(32)));
}

TEST_F(MLIRTypeHelperTest, widen_different_signedness_one_bit_gap_rejects)
{
    // MLIR integer types allow arbitrary bit widths, not just powers of two - si8 ->
    // ui9 has EXACTLY a 1-bit gap, which the ">1" check (line ~1913) deliberately
    // excludes, so this is the boundary case that must reject despite the gap rule.
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(si(8), ui(9)));
}

TEST_F(MLIRTypeHelperTest, widen_float_to_wider_float_accepts)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(f32(), f64()));
}

TEST_F(MLIRTypeHelperTest, widen_float_to_narrower_float_rejects)
{
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(f64(), f32()));
}

TEST_F(MLIRTypeHelperTest, widen_any_int_to_any_float_accepts_unconditionally)
{
    // No bit-width check at all for int->float, unlike float->float - a 64-bit int
    // widening "without data loss" to a 32-bit float is CURRENT behavior.
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(64), f32()));
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(8), f64()));
}

TEST_F(MLIRTypeHelperTest, widen_boolean_to_type_predicate_accepts)
{
    auto predType = mlir_ts::TypePredicateType::get(sym("p"), get<mlir_ts::BooleanType>(), false, 0);
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::BooleanType>(), predType));
}

TEST_F(MLIRTypeHelperTest, widen_boolean_to_integer_rejects)
{
    // explicitly NOT treated as an integer (see the commented-out block in the source)
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::BooleanType>(), si(32)));
}

TEST_F(MLIRTypeHelperTest, widen_literal_int_delegates_to_underlying_type)
{
    // literal(1: i32) -> number: widens exactly like a plain i32 would
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(literalInt(1, si(32)), get<mlir_ts::NumberType>()));
    // literal(1: i32) -> narrower i8: rejected exactly like a plain i32 would be
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(literalInt(1, si(32)), si(8)));
}

TEST_F(MLIRTypeHelperTest, widen_literal_int_to_index_when_within_size_bits)
{
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(literalInt(1, si(32)), get<mlir::IndexType>()));
}

TEST_F(MLIRTypeHelperTest, widen_enum_delegates_to_element_type)
{
    auto elementType = si(32);
    auto values = mlir::DictionaryAttr::get(getContext(), {});
    auto enumType = mlir_ts::EnumType::get(sym("E"), elementType, values);
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(elementType, enumType));
}

TEST_F(MLIRTypeHelperTest, widen_empty_const_array_to_any_array_accepts)
{
    auto emptyConstArray = mlir_ts::ConstArrayType::get(get<mlir_ts::NumberType>(), 0);
    auto stringArray = mlir_ts::ArrayType::get(get<mlir_ts::StringType>());
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(emptyConstArray, stringArray));
}

TEST_F(MLIRTypeHelperTest, widen_nonempty_const_array_checks_element_type)
{
    auto numberConstArray = mlir_ts::ConstArrayType::get(get<mlir_ts::NumberType>(), 3);
    auto numberArray = mlir_ts::ArrayType::get(get<mlir_ts::NumberType>());
    auto stringArray = mlir_ts::ArrayType::get(get<mlir_ts::StringType>());
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(numberConstArray, numberArray));
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(numberConstArray, stringArray));
}

TEST_F(MLIRTypeHelperTest, widen_undefined_to_optional_accepts)
{
    auto optNumber = mlir_ts::OptionalType::get(getContext(), get<mlir_ts::NumberType>());
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::UndefinedType>(), optNumber));
}

TEST_F(MLIRTypeHelperTest, widen_value_to_optional_of_compatible_type_accepts)
{
    auto optNumber = mlir_ts::OptionalType::get(getContext(), get<mlir_ts::NumberType>());
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(si(32), optNumber));
}

TEST_F(MLIRTypeHelperTest, widen_optional_to_optional_checks_element_types)
{
    auto optSi8 = mlir_ts::OptionalType::get(getContext(), si(8));
    auto optNumber = mlir_ts::OptionalType::get(getContext(), get<mlir_ts::NumberType>());
    EXPECT_TRUE(mth().canWideTypeWithoutDataLoss(optSi8, optNumber));
}

TEST_F(MLIRTypeHelperTest, widen_unrelated_types_reject)
{
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::StringType>(), get<mlir_ts::NumberType>()));
    EXPECT_FALSE(mth().canWideTypeWithoutDataLoss(get<mlir_ts::BooleanType>(), get<mlir_ts::StringType>()));
}

// ---------------------------------------------------------------------------
// isValueType / isNumericType
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, is_value_type_true_for_numeric_and_struct_like)
{
    EXPECT_TRUE(mth().isValueType(si(32)));
    EXPECT_TRUE(mth().isValueType(get<mlir_ts::NumberType>()));
    EXPECT_TRUE(mth().isValueType(get<mlir_ts::BooleanType>()));
    EXPECT_TRUE(mth().isValueType(tuple({})));
}

TEST_F(MLIRTypeHelperTest, is_value_type_false_for_reference_like)
{
    EXPECT_FALSE(mth().isValueType(get<mlir_ts::StringType>()));
    EXPECT_FALSE(mth().isValueType(mlir_ts::ArrayType::get(get<mlir_ts::NumberType>())));
}

TEST_F(MLIRTypeHelperTest, is_numeric_type)
{
    EXPECT_TRUE(mth().isNumericType(si(32)));
    EXPECT_TRUE(mth().isNumericType(get<mlir::IndexType>()));
    EXPECT_TRUE(mth().isNumericType(get<mlir_ts::NumberType>()));
    EXPECT_FALSE(mth().isNumericType(get<mlir_ts::StringType>()));
    EXPECT_FALSE(mth().isNumericType(get<mlir_ts::BooleanType>()));
    EXPECT_FALSE(mth().isNumericType(mlir::Type()));
}

// ---------------------------------------------------------------------------
// strip*Type / convert* - pure unwrap/rewrap helpers.
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, strip_literal_type_unwraps)
{
    EXPECT_EQ(mth().stripLiteralType(literalInt(1, si(32))), si(32));
}

TEST_F(MLIRTypeHelperTest, strip_literal_type_passthrough_for_non_literal)
{
    EXPECT_EQ(mth().stripLiteralType(si(32)), si(32));
}

TEST_F(MLIRTypeHelperTest, strip_optional_type_unwraps)
{
    auto opt = mlir_ts::OptionalType::get(getContext(), get<mlir_ts::NumberType>());
    EXPECT_EQ(mth().stripOptionalType(opt), get<mlir_ts::NumberType>());
}

TEST_F(MLIRTypeHelperTest, strip_ref_type_unwraps)
{
    auto ref = mlir_ts::RefType::get(get<mlir_ts::NumberType>());
    EXPECT_EQ(mth().stripRefType(ref), get<mlir_ts::NumberType>());
}

TEST_F(MLIRTypeHelperTest, convert_const_array_to_array_type)
{
    auto constArray = mlir_ts::ConstArrayType::get(get<mlir_ts::NumberType>(), 3);
    auto result = mth().convertConstArrayTypeToArrayType(constArray);
    EXPECT_EQ(result, mlir_ts::ArrayType::get(get<mlir_ts::NumberType>()));
}

TEST_F(MLIRTypeHelperTest, convert_const_array_passthrough_for_non_const_array)
{
    auto array = mlir_ts::ArrayType::get(get<mlir_ts::NumberType>());
    EXPECT_EQ(mth().convertConstArrayTypeToArrayType(array), array);
}

TEST_F(MLIRTypeHelperTest, convert_const_tuple_to_tuple_round_trip)
{
    llvm::SmallVector<mlir_ts::FieldInfo> fields{field(strAttr("x"), get<mlir_ts::NumberType>())};
    auto constTupleType = mlir_ts::ConstTupleType::get(getContext(), fields);
    auto tupleType = mth().convertConstTupleTypeToTupleType(constTupleType);
    EXPECT_EQ(tupleType.getFields().size(), 1u);
    EXPECT_EQ(tupleType.getFields()[0].id, strAttr("x"));

    auto backToConst = mth().convertTupleTypeToConstTupleType(tupleType);
    EXPECT_EQ(backToConst.getFields().size(), 1u);
}

// ---------------------------------------------------------------------------
// getFirstNonNullUnionType
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, first_non_null_union_type_skips_leading_null)
{
    auto unionType = mlir_ts::UnionType::get(getContext(), ::llvm::ArrayRef<mlir::Type>{get<mlir_ts::NullType>(), get<mlir_ts::NumberType>()});
    EXPECT_EQ(mth().getFirstNonNullUnionType(unionType), get<mlir_ts::NumberType>());
}

TEST_F(MLIRTypeHelperTest, first_non_null_union_type_all_null_returns_null_type)
{
    auto unionType = mlir_ts::UnionType::get(getContext(), ::llvm::ArrayRef<mlir::Type>{get<mlir_ts::NullType>()});
    EXPECT_FALSE(mth().getFirstNonNullUnionType(unionType));
}

// ---------------------------------------------------------------------------
// getUnionType (simple SmallVector overload)
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, union_type_single_element_returns_element_unwrapped)
{
    mlir::SmallVector<mlir::Type> types{get<mlir_ts::NumberType>()};
    EXPECT_EQ(mth().getUnionType(types), get<mlir_ts::NumberType>());
}

TEST_F(MLIRTypeHelperTest, union_type_empty_returns_empty_tuple)
{
    mlir::SmallVector<mlir::Type> types{};
    auto result = mth().getUnionType(types);
    auto asTuple = mlir::dyn_cast<mlir_ts::TupleType>(result);
    ASSERT_TRUE(static_cast<bool>(asTuple));
    EXPECT_EQ(asTuple.getFields().size(), 0u);
}

// ---------------------------------------------------------------------------
// canCastFromTo - representative cases (interface-related branches need a
// getInterfaceInfoByFullName callback, exercised elsewhere via full compiles).
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, cast_identical_types_accepts)
{
    EXPECT_TRUE(mth().canCastFromTo(loc(), si(32), si(32)));
}

TEST_F(MLIRTypeHelperTest, cast_widenable_types_accepts)
{
    EXPECT_TRUE(mth().canCastFromTo(loc(), si(32), get<mlir_ts::NumberType>()));
}

TEST_F(MLIRTypeHelperTest, cast_unrelated_types_rejects)
{
    EXPECT_FALSE(mth().canCastFromTo(loc(), get<mlir_ts::StringType>(), get<mlir_ts::NumberType>()));
}

// KNOWN BUG, characterized here rather than fixed (see canCastFromTo, the
// `dyn_cast<mlir_ts::UnionType>(destType)` branch, ~line 1555): the find_if
// predicate direction is backwards (checks `item -> srcType` instead of the
// intended `srcType -> item`), AND even when a member match IS found, the `if`
// block only returns early on the NOT-found case - the found case falls
// through to the function's unconditional trailing `return false`. So this
// branch can structurally never contribute a `true` result, for ANY srcType/
// destType pair, not just this one. Confirmed NOT observable end-to-end: both
// `let x: number | string = "hello"` and `<number | string>s` compile and run
// correctly, so real union-assignment/cast validation must route through a
// different mechanism (see the neighboring "TODO: we have Cast verification
// which does not have connected getInterfaceInfoByFullName" comment) - this
// dead branch may be a genuinely unreachable vestige rather than a live bug,
// but a caller relying on canCastFromTo specifically for union-destination
// validation would silently always get `false`. Left as a documented
// characterization test (not fixed) pending a fuller audit of every
// canCastFromTo call site with a union destType.
TEST_F(MLIRTypeHelperTest, cast_into_union_currently_always_rejects_even_on_member_match)
{
    auto unionType = mlir_ts::UnionType::get(getContext(), ::llvm::ArrayRef<mlir::Type>{get<mlir_ts::NumberType>(), get<mlir_ts::StringType>()});
    EXPECT_FALSE(mth().canCastFromTo(loc(), get<mlir_ts::StringType>(), unionType));
}

TEST_F(MLIRTypeHelperTest, cast_into_union_rejects_if_no_member_matches)
{
    auto unionType = mlir_ts::UnionType::get(getContext(), ::llvm::ArrayRef<mlir::Type>{get<mlir_ts::NumberType>(), get<mlir_ts::BooleanType>()});
    EXPECT_FALSE(mth().canCastFromTo(loc(), get<mlir_ts::StringType>(), unionType));
}

TEST_F(MLIRTypeHelperTest, cast_tuple_to_identically_shaped_tuple_accepts)
{
    auto srcTuple = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    auto dstTuple = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    EXPECT_TRUE(mth().canCastFromTo(loc(), srcTuple, dstTuple));
}

TEST_F(MLIRTypeHelperTest, cast_tuple_to_tuple_requires_size_equal_fields_not_just_widenable)
{
    // canCastFromToLogic (tuple/tuple branch) uses isSizeEqual per field, NOT
    // canWideTypeWithoutDataLoss - a tuple-to-tuple cast is a structural/storage-layout
    // reinterpretation, not a value conversion, so a merely-WIDENABLE (but
    // differently-sized) field type is correctly rejected here even though
    // canWideTypeWithoutDataLoss(si32, number) alone would accept it.
    auto srcTuple = tuple({field(strAttr("x"), si(32))});
    auto dstTuple = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    EXPECT_FALSE(mth().canCastFromTo(loc(), srcTuple, dstTuple));
}

TEST_F(MLIRTypeHelperTest, cast_tuple_to_tuple_rejects_field_count_mismatch)
{
    auto srcTuple = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    auto dstTuple = tuple({field(strAttr("x"), get<mlir_ts::NumberType>()), field(strAttr("y"), get<mlir_ts::NumberType>())});
    EXPECT_FALSE(mth().canCastFromTo(loc(), srcTuple, dstTuple));
}

// ---------------------------------------------------------------------------
// mergeType
// ---------------------------------------------------------------------------

TEST_F(MLIRTypeHelperTest, merge_type_identical_reports_merged)
{
    bool merged = false;
    auto result = mth().mergeType(loc(), si(32), si(32), merged);
    EXPECT_TRUE(merged);
    EXPECT_EQ(result, si(32));
}

TEST_F(MLIRTypeHelperTest, merge_type_widenable_picks_wider)
{
    bool merged = false;
    auto result = mth().mergeType(loc(), si(32), get<mlir_ts::NumberType>(), merged);
    EXPECT_TRUE(merged);
    EXPECT_EQ(result, get<mlir_ts::NumberType>());
}

// ---------------------------------------------------------------------------
// MLIRCodeLogic.h: TupleFieldGetterAndSetter / TupleFieldTypeNoError - the
// mangled get_x/set_x lookup and field-by-id-or-index resolution that
// DeclarationPrinter's accessor-exclusion logic (see
// declaration-printer-unit-tests) sits downstream of.
// ---------------------------------------------------------------------------

struct MLIRCodeLogicTest : public MLIRTypeHelperTest
{
    MLIRCodeLogic mcl()
    {
        return MLIRCodeLogic(getContext(), compileOptions);
    }
};

TEST_F(MLIRCodeLogicTest, tuple_field_getter_and_setter_found)
{
    auto t = tuple({
        field(strAttr("get_celsius"), get<mlir_ts::NumberType>()),
        field(strAttr("set_celsius"), get<mlir_ts::NumberType>()),
    });
    auto [getterIndex, setterIndex] = mcl().TupleFieldGetterAndSetter(t, strAttr("celsius"));
    EXPECT_EQ(getterIndex, 0);
    EXPECT_EQ(setterIndex, 1);
}

TEST_F(MLIRCodeLogicTest, tuple_field_getter_and_setter_not_found)
{
    auto t = tuple({field(strAttr("plain"), get<mlir_ts::NumberType>())});
    auto [getterIndex, setterIndex] = mcl().TupleFieldGetterAndSetter(t, strAttr("celsius"));
    EXPECT_EQ(getterIndex, -1);
    EXPECT_EQ(setterIndex, -1);
}

TEST_F(MLIRCodeLogicTest, tuple_field_getter_and_setter_requires_exact_name_match)
{
    // "get_celsius2" must not match a lookup for "celsius" (prefix/suffix trap:
    // starts_with("get_") && ends_with(fieldName) alone isn't enough - the size
    // check guards against a longer field name that merely ends with the target)
    auto t = tuple({field(strAttr("get_celsius2"), get<mlir_ts::NumberType>())});
    auto [getterIndex, setterIndex] = mcl().TupleFieldGetterAndSetter(t, strAttr("celsius"));
    EXPECT_EQ(getterIndex, -1);
    EXPECT_EQ(setterIndex, -1);
}

TEST_F(MLIRCodeLogicTest, tuple_field_type_no_error_by_name)
{
    auto t = tuple({field(strAttr("x"), get<mlir_ts::NumberType>()), field(strAttr("y"), get<mlir_ts::StringType>())});
    auto [index, type, access] = mcl().TupleFieldTypeNoError(t, strAttr("y"));
    EXPECT_EQ(index, 1);
    EXPECT_EQ(type, get<mlir_ts::StringType>());
    EXPECT_EQ(access, mlir_ts::AccessLevel::Public);
}

TEST_F(MLIRCodeLogicTest, tuple_field_type_no_error_not_found)
{
    auto t = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    auto [index, type, access] = mcl().TupleFieldTypeNoError(t, strAttr("missing"));
    EXPECT_EQ(index, -1);
    EXPECT_FALSE(type);
}

TEST_F(MLIRCodeLogicTest, tuple_field_type_no_error_by_index_access)
{
    auto t = tuple({field(strAttr("x"), get<mlir_ts::NumberType>()), field(strAttr("y"), get<mlir_ts::StringType>())});
    auto intAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 1);
    auto [index, type, access] = mcl().TupleFieldTypeNoError(t, intAttr, /*indexAccess*/ true);
    EXPECT_EQ(index, 1);
    EXPECT_EQ(type, get<mlir_ts::StringType>());
}

TEST_F(MLIRCodeLogicTest, tuple_field_type_no_error_index_out_of_bounds)
{
    auto t = tuple({field(strAttr("x"), get<mlir_ts::NumberType>())});
    auto intAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), 5);
    auto [index, type, access] = mcl().TupleFieldTypeNoError(t, intAttr, /*indexAccess*/ true);
    EXPECT_EQ(index, -1);
}

TEST_F(MLIRCodeLogicTest, tuple_field_type_preserves_access_level)
{
    auto t = tuple({field(strAttr("secret"), get<mlir_ts::NumberType>(), false, mlir_ts::AccessLevel::Private)});
    auto [index, type, access] = mcl().TupleFieldTypeNoError(t, strAttr("secret"));
    EXPECT_EQ(access, mlir_ts::AccessLevel::Private);
}
