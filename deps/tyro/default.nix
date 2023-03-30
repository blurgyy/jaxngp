# with import <nixpkgs> {};

{ source, lib, buildPythonPackage

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

  meta = {
    homepage = "https://github.com/brentyi/tyro";
    description = "Strongly typed, zero-effort CLI interfaces & config objects";
    license = lib.licenses.mit;
  };
}
