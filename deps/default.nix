{ pkgs, lib }:

pypkgs: let
  mapDir = basedir: fn: with builtins;
    mapAttrs (name: _: fn name)
      (lib.filterAttrs
        (_: type: type == "directory")
        (readDir basedir));
  mkPyPackage = name: let
      generated = pkgs.callPackage ./_sources/generated.nix {};
    in pypkgs.callPackage ./${name} {
        source = generated.${name};
      };
in
  mapDir ./. mkPyPackage
