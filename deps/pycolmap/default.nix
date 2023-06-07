{ source, config, lib, buildPythonPackage

, cmake
, setuptools-scm

, cudatoolkit
, boost17x
, ceres-solver
, colmap-locked
, eigen
, flann
, freeimage
, libGLU
, metis
, glew
, qt5
}:

let
  cudaSupport = config.cudaSupport or false;
in

buildPythonPackage {
  pname = if cudaSupport
    then source.pname + "-cuda"
    else source.pname;
  inherit (source) version src;

  nativeBuildInputs = [
    cmake
    setuptools-scm
    qt5.wrapQtAppsHook
  ];
  dontUseCmakeConfigure = true;
  cmakeFlags = [
    "-DCUDA_ENABLED=ON"
    "-DCUDA_NVCC_FLAGS=--std=c++14"
  ];

  buildInputs = [
    colmap-locked
    cudatoolkit
    boost17x
    ceres-solver
    eigen
    flann
    freeimage
    libGLU
    metis
    glew
    qt5.qtbase
  ];

  preBuild = ''
    export MAKEFLAGS="''${MAKEFLAGS:+''${MAKEFLAGS} }-j$NIX_BUILD_CORES"
  '';

  meta = {
    homepage = "https://github.com/colmap/pycolmap";
    description = "Python bindings for COLMAP";
    license = lib.licenses.bsd3;
  };
}
