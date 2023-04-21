{ source, config, lib, buildPythonPackage

, cmake
, setuptools-scm

, cudatoolkit
, boost17x
, ceres-solver
, colmap
, eigen
, flann
, freeimage
, libGLU
, metis
, glew
, qt5
}:

buildPythonPackage {
  pname = if config.cudaSupport or false
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
    colmap
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

  preBuild = "export MAKEFLAGS=-j$NIX_BUILD_CORES";

  meta = {
    homepage = "https://github.com/colmap/pycolmap";
    description = "Python bindings for COLMAP";
    license = lib.licenses.bsd3;
  };
}
