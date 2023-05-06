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
  # Compute capability, used for building tiny-cuda-nn
  # REF: <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>
  cudaCapabilities = [ "6.1" ];  # 1080Ti
in {
  inherit filterAttrs;
  packages = pkgs: mapPackage ./. (name: pkgs.${name});
  overlay = final: prev: mapPackage ./. (name: let
    generated = final.callPackage ./_sources/generated.nix {};
    package = import ./${name};
    args = with builtins; intersectAttrs (functionArgs package) {
      inherit generated cudaCapabilities;
      version = "0.1.0";
      source = generated.${name};
      buildSharedLib = false;
    };
  in
    final.python3.pkgs.callPackage package args
  );
}
