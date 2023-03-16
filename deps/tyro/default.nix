# with import <nixpkgs> {};

{ fetchPypi, buildPythonPackage
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

buildPythonPackage rec {
  pname = "tyro";
  version = "0.4.2";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-8UaFs/mMsQKOOiuSpNrpvkRA0YmXDqTi7IIN3TV/Ti4=";
  };
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
