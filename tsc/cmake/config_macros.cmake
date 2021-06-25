macro(set_Options)

if(MSVC)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch -Wno-unused-function -Wno-unused-result -Wno-sign-compare -Wno-implicit-fallthrough -Wno-logical-op-parentheses -lstdc++fs")
endif()

endmacro()


