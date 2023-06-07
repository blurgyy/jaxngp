{ source, lib, config, colmap
, flann
, metis
}:

let
  cudaSupport = config.cudaSupport or false;
in

colmap.overrideAttrs (o: {
  pname = if cudaSupport
    then source.pname + "-cuda"
    else source.pname;
  inherit (source) version src;
  buildInputs = o.buildInputs ++ [
    flann
    metis
  ];
  cmakeFlags = o.cmakeFlags ++ (lib.optional
    cudaSupport
    "-DCMAKE_CUDA_ARCHITECTURES=all-major"
  );
})
