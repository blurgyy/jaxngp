let
  filterAttrs = predicate: attrs: with builtins; listToAttrs (filter
    (v: v != null)
    (attrValues (mapAttrs
      (name: value: if (predicate name value) then { inherit name value; } else null)
      attrs)
    )
  );
  mapPackage = basedir: fn: with builtins;
    mapAttrs (name: _: fn name)
      (filterAttrs
        (name: type: type == "directory" && name != "_sources")
        (readDir basedir));
in {
  inherit filterAttrs;
  packages = pkgs: mapPackage ./. (name: pkgs.${name});
  overlay = final: prev: mapPackage ./. (name: let
    generated = final.callPackage ./_sources/generated.nix {};
    package = import ./${name};
    args = with builtins; intersectAttrs (functionArgs package) {
      inherit generated;
      source = generated.${name};
    };
  in
    final.python3.pkgs.callPackage package args
  );
}
