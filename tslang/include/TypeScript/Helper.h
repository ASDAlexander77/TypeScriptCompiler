#ifndef TYPESCRIPT_HELPER_H
#define TYPESCRIPT_HELPER_H

#include <memory>

template< typename T, typename U >
inline std::unique_ptr< T > dynamic_pointer_cast(std::unique_ptr< U > &&ptr) {
    U * const stored_ptr = ptr.release();
    T * const converted_stored_ptr = dynamic_cast< T * >(stored_ptr);
    if (converted_stored_ptr) {
        return std::unique_ptr< T >(converted_stored_ptr);
    }
    else {
        if (stored_ptr)
        {
            throw "Invalid cast";
        }

        ptr.reset(stored_ptr);
        return std::unique_ptr< T >();
    }
}

#endif // TYPESCRIPT_HELPER_H