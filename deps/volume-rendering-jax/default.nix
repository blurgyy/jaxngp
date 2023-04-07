{ symlinkJoin, buildPythonPackage

, setuptools-scm
, cmake
, ninja
, pybind11
, fmt

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
in

buildPythonPackage {
  pname = "volume-rendering-jax";
  version = "0.1.0";
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
    cudatoolkit-unsplit
    fmt.dev
  ];

  propagatedBuildInputs = [
    chex
    jax
    jaxlib
  ];

  doCheck = false;

  pythonImportsCheck = [ "volrendjax" ];

  # development
  dot_clangd = ''
    CompileFlags:                     # Tweak the parse settings
      Add:
        - "-Wall"                     # enable more warnings
        - "-std=c++20"                # use cpp20 standard (std::bit_cast needs this)
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
