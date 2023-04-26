{ lib, version, symlinkJoin, linkFarm, cudaCapabilities, buildPythonPackage

, setuptools-scm
, cmake
, ninja
, pybind11
, fmt

, serde-helper
, cudatoolkit
, tiny-cuda-nn
, nlohmann_json
, python3
, chex
, jax
, jaxlib
}:

let
  dropDot = x: builtins.replaceStrings ["."] [""] x;
  minGpuArch = let
    min = lhs: rhs: if (builtins.compareVersions lhs rhs) < 0
      then lhs
      else rhs;
  in dropDot (builtins.foldl' min "998244353" cudaCapabilities);

  cudatoolkit-unsplit = symlinkJoin {
    name = "${cudatoolkit.name}-unsplit";
    paths = [ cudatoolkit.out cudatoolkit.lib ];
  };
  fmt-unsplit = symlinkJoin {
    name = "${fmt.name}-unsplit";
    paths = [ fmt.out fmt.dev ];
  };
  nlohmann_json-symlinked = linkFarm "${nlohmann_json.name}-symlinked" [
    { name = "include/json"; path = "${nlohmann_json}/include/nlohmann"; }
    { name = "include/nlohmann"; path = "${nlohmann_json}/include/nlohmann"; }
  ];
in

buildPythonPackage rec {
  pname = "jax-tcnn";
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
  cmakeFlags = [
    "-DTCNN_MIN_GPU_ARCH=${minGpuArch}"
    "-DCMAKE_CUDA_ARCHITECTURES=${lib.concatStringsSep ";" (map dropDot cudaCapabilities)}"
  ];

  buildInputs = [
    cudatoolkit-unsplit
    fmt-unsplit
    serde-helper
    tiny-cuda-nn
    nlohmann_json-symlinked
  ];

  propagatedBuildInputs = [
    chex
    jax
    jaxlib
  ];

  preFixup = ''
    patchelf --set-rpath "${lib.makeLibraryPath buildInputs}/lib" $out/lib/python${python3.pythonVersion}/site-packages/jaxtcnn/*.so
  '';

  doCheck = false;

  pythonImportsCheck = [ "jaxtcnn" ];

  # development
  dot_clangd = ''
    CompileFlags:                     # Tweak the parse settings
      Add:
        - "-Wall"                     # enable more warnings
        - "-Wshadow"                  # warn if a local declared variable shadows a global one
        - "-std=c++20"                # use cpp20 standard (std::bit_cast needs this)
        - "-DTCNN_MIN_GPU_ARCH=${minGpuArch}"
        - "-I${tiny-cuda-nn}/include"
        - "-I${nlohmann_json-symlinked}/include"
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
