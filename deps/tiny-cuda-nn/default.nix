# adapted from <https://github.com/NixOS/nixpkgs/blob/aacbb6d021d617881630c27088f311a5bc0e2145/pkgs/development/libraries/science/math/tiny-cuda-nn/default.nix>

{ source, cudaCapabilities

, buildSharedLib ? false

, cmake
, cudaPackages
, lib
, ninja
, stdenv
, symlinkJoin
, which
}:
let
  cuda-common-redist = with cudaPackages; [
    libcublas # cublas_v2.h
    libcusolver # cusolverDn.h
    libcusparse # cusparse.h
  ];

  cuda-native-redist = symlinkJoin {
    name = "cuda-redist";
    paths = with cudaPackages; [
      cuda_cudart # cuda_runtime.h
      cuda_nvcc
    ] ++ cuda-common-redist;
  };

  cuda-redist = symlinkJoin {
    name = "cuda-redist";
    paths = cuda-common-redist;
  };
in
stdenv.mkDerivation (finalAttrs: rec {
  inherit (source) pname version src;

  outputs = [ "out" "dev" ];

  nativeBuildInputs = [
    cmake
    cuda-native-redist
    ninja
    which
  ];

  # build a shared library for faster development
  postPatch = lib.optionalString buildSharedLib ''
    sed -E \
      -e 's/BUILD_SHARED_LIBS OFF/BUILD_SHARED_LIBS ON/g' \
      -e 's/STATIC/SHARED/g' \
      -i CMakeLists.txt
  '';

  # by default tcnn builds a static library, but that's too slow
  cmakeFlags = [
    "-DTCNN_BUILD_EXAMPLES=OFF"
    "-DTCNN_BUILD_BENCHMARK=OFF"
  ];

  buildInputs = [
    cuda-redist
  ];

  # NOTE: We cannot use pythonImportsCheck for this module because it uses torch to immediately
  #   initailize CUDA and GPU access is not allowed in the nix build environment.
  # NOTE: There are no tests for the C++ library or the python bindings, so we just skip the check
  #   phase -- we're not missing anything.
  doCheck = false;

  preConfigure = let
    dropDot = x: builtins.replaceStrings ["."] [""] x;
  in ''
    export TCNN_CUDA_ARCHITECTURES=${
      lib.concatStringsSep "\\;" (map dropDot cudaCapabilities)
    }
    export CUDA_HOME=${cuda-native-redist}
    export LIBRARY_PATH=${cuda-native-redist}/lib/stubs:$LIBRARY_PATH
  '';

  installPhase = ''
    runHook preInstall

    # install headers
    mkdir -p $dev/include
    cp -vr ../include/* $dev/include
    cp -vr ../dependencies/* $dev/include

    # install built library
    mkdir -p $out/lib $out
    cp -v libtiny-cuda-nn.${if buildSharedLib then "so" else "a"} $out/lib/

    runHook postInstall
  '';
  # Fixes:
  #   > RPATH of binary /nix/store/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-tiny-cuda-nn-v1.6/lib/libtiny-cuda-nn.so contains a forbidden reference to /build/
  # REF: <https://github.com/calvertvl/nixpkgs/commit/6798060d8f5116a7d329df3265e82e0645aefc8f>
  preFixup = lib.optionalString buildSharedLib ''
    patchelf --set-rpath ${lib.makeLibraryPath buildInputs} $out/lib/libtiny-cuda-nn.so
  '';

  passthru = {
    inherit cudaPackages;
  };

  meta = with lib; {
    description = "Lightning fast C++/CUDA neural network framework";
    homepage = "https://github.com/NVlabs/tiny-cuda-nn";
    license = licenses.bsd3;
    maintainers = with maintainers; [ connorbaker ];
    platforms = platforms.linux;
  };
})
