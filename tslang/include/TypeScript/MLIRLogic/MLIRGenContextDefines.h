namespace
{
#define V(x) static_cast<mlir::Value>(x)

#define CAST(res_cast, location, to_type, from_value, gen_context) \
    auto cast_result = cast(location, to_type, from_value, gen_context); \
    EXIT_IF_FAILED_OR_NO_VALUE(cast_result) \
    res_cast = V(cast_result);

#define CAST_A(res_cast, location, to_type, from_value, gen_context) \
    mlir::Value res_cast; \
    { \
        auto cast_result = cast(location, to_type, from_value, gen_context); \
        EXIT_IF_FAILED_OR_NO_VALUE(cast_result) \
        res_cast = V(cast_result); \
    }

#define CAST_A_NULLCHECK(res_cast, location, to_type, from_value, gen_context, nullcheck) \
    mlir::Value res_cast; \
    { \
        auto cast_result = cast(location, to_type, from_value, gen_context, nullcheck); \
        EXIT_IF_FAILED_OR_NO_VALUE(cast_result) \
        res_cast = V(cast_result); \
    }

#define DECLARE(varDesc, varValue) \
    if (mlir::failed(declare(location, varDesc, varValue, genContext))) \
    { \
        return mlir::failure(); \
    }

} // namespace
