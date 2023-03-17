# with import <nixpkgs> {};

{ source, buildPythonPackage

, poetry-core
, backports-cached-property

, colorama
, frozendict
, pyyaml
, typing-extensions

, docstring-parser
, rich
, shtab
}:

buildPythonPackage {
  inherit (source) pname version src;
  format = "pyproject";

  nativeBuildInputs = [
    poetry-core
    backports-cached-property
  ];

  buildInputs = [
    colorama
    frozendict
    pyyaml
    typing-extensions
  ];

  propagatedBuildInputs = [
    docstring-parser
    rich
    shtab
  ];

  pythonImportsCheck = [ "tyro" ];
}
