{ source, lib, buildPythonPackage

, setuptools-scm

, click
, cython
, glfw
, pyopengl
, wheel
}:

buildPythonPackage {
  inherit (source) pname version src;
  format = "pyproject";

  nativeBuildInputs = [ setuptools-scm ];
  buildInputs = [ wheel ];
  propagatedBuildInputs = [ click cython glfw pyopengl ];

  pythonImportsCheck = [ "imgui" ];

  meta = {
    homepage = "https://github.com/pyimgui/pyimgui";
    description = "Cython-based Python bindings for dear imgui";
    license = lib.licenses.bsd3;
  };
}
