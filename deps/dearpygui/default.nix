{ source, lib, buildPythonPackage

, cmake
, setuptools-scm

, libglvnd
, libxcrypt
, xorg

, pillow
}:

buildPythonPackage {
  inherit (source) pname version src;

  nativeBuildInputs = [
    cmake
    setuptools-scm
  ];
  dontUseCmakeConfigure = true;

  preBuild = ''
    export MAKEFLAGS="''${MAKEFLAGS:+''${MAKEFLAGS} }-j$NIX_BUILD_CORES"
  '';

  buildInputs = with xorg; [
    libX11
    libXcursor
    libXi
    libXinerama
    libXrandr
  ] ++ [
    libglvnd
    libxcrypt
  ];

  propagatedBuildInputs = [ pillow ];

  doCheck = false;

  pythonImportsCheck = [
    "dearpygui"
    "dearpygui.dearpygui"
  ];

  meta = {
    homepage = "https://github.com/hoffstadt/DearPyGui";
    description = "A fast and powerful Graphical User Interface Toolkit for Python with minimal dependencies";
    license = lib.licenses.mit;
  };
}
