# SPDX-License-Identifier: MIT

# Resolve the ONNX Runtime C API headers without linking the runtime. The Python
# environment and deployment image provide the DLL/shared object; this project only
# needs the ABI declarations while compiling the dynamically loaded backend.
function(parkfit_resolve_onnxruntime_headers out_var)
  set(PARKFIT_ONNXRUNTIME_VERSION "1.29.0" CACHE STRING
      "ONNX Runtime C API header version")
  set(PARKFIT_ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH
      "Existing directory containing ONNX Runtime C API headers")

  if(PARKFIT_ONNXRUNTIME_INCLUDE_DIR)
    if(NOT EXISTS "${PARKFIT_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
      message(FATAL_ERROR
        "PARKFIT_ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_c_api.h: "
        "${PARKFIT_ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    set(${out_var} "${PARKFIT_ONNXRUNTIME_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  set(_ort_include "${CMAKE_BINARY_DIR}/_deps/onnxruntime-${PARKFIT_ONNXRUNTIME_VERSION}/include")
  set(_ort_base
      "https://raw.githubusercontent.com/microsoft/onnxruntime/v${PARKFIT_ONNXRUNTIME_VERSION}/include/onnxruntime/core/session")
  set(_ort_headers
      "onnxruntime_c_api.h|ACC0CF4B3F28D39339C76770D76164BB7A0637DC89F5FDE764B4017B632F6743"
      "onnxruntime_ep_c_api.h|E6C986C9E98583F8113B2C6BC3864814883B806D501CF24DA4D239C45753E235"
      "onnxruntime_error_code.h|5CE3B054E798ECED8D14F5B86E98692FD33470463F96194CE0700A2D53DD8721")

  file(MAKE_DIRECTORY "${_ort_include}")
  foreach(_entry IN LISTS _ort_headers)
    string(REPLACE "|" ";" _parts "${_entry}")
    list(GET _parts 0 _name)
    list(GET _parts 1 _sha256)
    set(_destination "${_ort_include}/${_name}")

    if(EXISTS "${_destination}")
      file(SHA256 "${_destination}" _actual_sha256)
      string(TOUPPER "${_actual_sha256}" _actual_sha256)
      if(NOT _actual_sha256 STREQUAL _sha256)
        message(STATUS "parkfit: replacing invalid cached ONNX Runtime header ${_name}")
        file(REMOVE "${_destination}")
      endif()
    endif()

    if(NOT EXISTS "${_destination}")
      message(STATUS "parkfit: fetching ONNX Runtime ${PARKFIT_ONNXRUNTIME_VERSION} ${_name}")
      file(DOWNLOAD
        "${_ort_base}/${_name}"
        "${_destination}"
        EXPECTED_HASH "SHA256=${_sha256}"
        TLS_VERIFY ON
        STATUS _download_status)
      list(GET _download_status 0 _download_code)
      list(GET _download_status 1 _download_message)
      if(NOT _download_code EQUAL 0)
        file(REMOVE "${_destination}")
        message(FATAL_ERROR "Failed to fetch ${_name}: ${_download_message}")
      endif()
    endif()
  endforeach()

  set(${out_var} "${_ort_include}" PARENT_SCOPE)
endfunction()
