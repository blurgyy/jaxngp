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
  # NOTE: removing unused compute capabilities will build faster, GPUs and their compute
  # capabilities can be found at:
  #   <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>
  # All the compute cababilities since `5.0`. REF: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability>
  cudaCapabilities = [
    # /nix/store/hsbbv8a72hwjrka8igd7hk66skvi03rp-cudatoolkit-11.7.0-unsplit/bin/nvcc --list-gpu-arch
    "3.5"
    "3.7"
    "5.0"
    "5.2"
    "5.3"
    "6.0"
    "6.1"
    "6.2"
    "7.0"
    "7.2"
    "7.5"
    "8.0"
    "8.6"
    "8.7"
    ## the two compute capabilities below require newer nvcc (this environment uses CUDA 11.7)
    # "8.9"
    # "9.0"
  ];
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
