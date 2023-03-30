{ pkgs, lib }:

pypkgs: let
  mapDir = basedir: fn: with builtins;
    mapAttrs (name: _: fn name)
      (lib.filterAttrs
        (_: type: type == "directory")
        (readDir basedir));
  mkPyPackage = name: let
      generated = pkgs.callPackage ./_sources/generated.nix {};
      package = import ./${name};
      args = with builtins; intersectAttrs (functionArgs package) {
        source = generated.${name};
      };
    in pypkgs.callPackage package args;
in
  mapDir ./. mkPyPackage
