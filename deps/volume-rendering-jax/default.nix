{ lib, version, symlinkJoin, buildPythonPackage

, setuptools-scm
, cmake
, ninja
, pybind11
, fmt

, serde-helper
, cudatoolkit
, python3
, chex
, jax
, jaxlib
}:

let
  cudatoolkit-unsplit = symlinkJoin {
    name = "${cudatoolkit.name}-unsplit";
    paths = [ cudatoolkit.out cudatoolkit.lib ];
  };
  fmt-unsplit = symlinkJoin {
    name = "fmtlib";
    # libfmt.so resides in the "out" output and is set into RPATH of the python extension
    paths = [ fmt.dev fmt.out ];
  };
in

buildPythonPackage rec {
  pname = "volume-rendering-jax";
  inherit version;
  src = ./.;

  format = "pyproject";

  CUDA_HOME = cudatoolkit-unsplit;

  nativeBuildInputs = [
    cmake
    ninja
    pybind11
    setuptools-scm
  ];
  dontUseCmakeConfigure = true;

  buildInputs = [
    serde-helper
    cudatoolkit-unsplit
    fmt-unsplit
  ];

  propagatedBuildInputs = [
    chex
    jax
    jaxlib
  ];

  preFixup = ''
    patchelf --set-rpath "${lib.makeLibraryPath buildInputs}" $out/lib/python${python3.pythonVersion}/site-packages/volrendjax/*.so
  '';

  doCheck = false;

  pythonImportsCheck = [ "volrendjax" ];

  # development
  dot_clangd = ''
    CompileFlags:                     # Tweak the parse settings
      Add:
        - "-Wall"                     # enable more warnings
        - "-Wshadow"                  # warn if a local declared variable shadows a global one
        - "-std=c++20"                # use cpp20 standard (std::bit_cast needs this)
        - "-I${serde-helper}/include"
        - "-I${cudatoolkit-unsplit}/include"
        - "-I${fmt.dev}/include"
        - "-I${pybind11}/include"
        - "-I${python3}/include/python${python3.pythonVersion}"
        - "--cuda-path=${cudatoolkit-unsplit}"
      Remove: "-W*"                   # strip all other warning-related flags
      Compiler: "clang++"             # Change argv[0] of compile flags to clang++

    # vim: ft=yaml:
  '';
  shellHook = ''
    echo "use \`echo \$dot_clangd >.clangd\` for development"
    [[ "$-" == *i* ]] && exec "$SHELL"
  '';
}
