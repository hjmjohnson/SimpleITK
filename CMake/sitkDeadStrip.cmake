#
# Discard unreferenced functions and data and fold identical functions at
# link time. Intended for distribution builds; GCC and Clang on ELF only.
#
option(
  SimpleITK_DEAD_STRIP
  "Discard unreferenced code and data and fold identical functions at link time (GCC/Clang on ELF)."
  OFF
)
mark_as_advanced(SimpleITK_DEAD_STRIP)

if(SimpleITK_DEAD_STRIP AND NOT MSVC AND NOT APPLE)
  include(sitkUseGoldLinker)

  foreach(_lang C CXX)
    string(
      STRIP
      "${CMAKE_${_lang}_FLAGS} -ffunction-sections -fdata-sections"
      CMAKE_${_lang}_FLAGS
    )
  endforeach()

  set(_sitk_dead_strip_flags "-Wl,--gc-sections")
  if(USE_GOLD_LINKER)
    set(_sitk_have_icf_safe 1)
  else()
    include(CheckLinkerFlag)
    check_linker_flag(CXX "-Wl,--icf=safe" _sitk_have_icf_safe)
  endif()
  if(_sitk_have_icf_safe)
    string(APPEND _sitk_dead_strip_flags " -Wl,--icf=safe")
  endif()

  foreach(_kind EXE MODULE SHARED)
    string(
      STRIP
      "${CMAKE_${_kind}_LINKER_FLAGS} ${_sitk_dead_strip_flags}"
      CMAKE_${_kind}_LINKER_FLAGS
    )
  endforeach()
endif()
