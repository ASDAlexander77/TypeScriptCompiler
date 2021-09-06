include_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc-8.0.4/include/")

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
link_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc/${CMAKE_BUILD_TYPE_LOWERCASE}/")

add_mlir_library(TypeScriptGCWrapper
  SHARED
  TypeScriptGC.cpp
  gc.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PRIVATE
  gcmt-lib
)
