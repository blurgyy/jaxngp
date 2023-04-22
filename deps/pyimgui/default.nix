{ source, lib, buildPythonPackage

, setuptools-scm

, cython
, pyopengl
, glfw
, wheel
, click
}:

buildPythonPackage {
  inherit (source) pname version src;
  format = "pyproject";

  nativeBuildInputs = [ setuptools-scm ];
  buildInputs = [
    click
    cython
    glfw
    pyopengl
    wheel
  ];

  pythonImportsCheck = [ "imgui" ];

  meta = {
    homepage = "https://github.com/pyimgui/pyimgui";
    description = "Cython-based Python bindings for dear imgui";
    license = lib.licenses.bsd3;
  };
}
