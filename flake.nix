{
  description = "Instant Neural Graphics Primitivs in JAX";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/22.11";
    nixpkgs-with-nvidia-driver-fix.url = "github:nixos/nixpkgs/pull/222762/head";
    flake-utils.url = "github:numtide/flake-utils/3db36a8b464d0c4532ba1c7dda728f4576d6d073";
    nixgl = {
      url = "github:guibou/nixgl/c917918ab9ebeee27b0dd657263d3f57ba6bb8ad";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: let
    inherit (nixpkgs) lib;
    basePkgs = import nixpkgs { inherit system; };
  in {
    devShells = let
      pyVer = "310";
      py = "python${pyVer}";
      depsWith = import ./deps { inherit lib; pkgs = basePkgs; };
      jaxOverlays = final: prev: {
          # avoid rebuilding opencv4 with cuda for tensorflow-datasets
          opencv4 = prev.opencv4.override {
              enableCuda = false;
            };
          ${py} = prev.${py}.override {
            packageOverrides = finalScope: prevScope: {
              jax = prevScope.jax.overridePythonAttrs (o: { doCheck = false; });
              jaxlib = prevScope.jaxlib-bin;
              flax = prevScope.flax.overridePythonAttrs (o: {
                  buildInputs = o.buildInputs ++ [ prevScope.pyyaml ];
                  doCheck = false;
                });
              tensorflow = prevScope.tensorflow.override {
                  # we only use tensorflow-datasets for data loading, it does not need to be built
                  # with cuda support (building with cuda support is too demanding).
                  cudaSupport = false;
                };
            };
          };
        };
        cudaPkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            packageOverrides = pkgs: {
              linuxPackages = (import inputs.nixpkgs-with-nvidia-driver-fix {}).linuxPackages;
            };
          };
          overlays = [
            inputs.nixgl.overlays.default
            jaxOverlays
          ];
        };
      cpuPkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = false;  # NOTE: disable cuda for cpu env
        };
        overlays = [
          inputs.nixgl.overlays.default
          jaxOverlays
        ];
      };
      mkPythonDeps = { pp, extraPackages }: with pp; [
          ipython
          tensorflow
          tqdm
          icecream
          (depsWith pp).dearpygui
          (depsWith pp).tyro
          pillow
          ipdb
          colorama

          jaxlib-bin
          jax
          optax
          flax
        ] ++ extraPackages;
    in rec {
      default = cudaDevShell;
      cudaDevShell = cudaPkgs.mkShell {  # impure
        name = "cuda";
        buildInputs = [
          cudaPkgs.colmapWithCuda
          (cudaPkgs.${py}.withPackages (pp: mkPythonDeps {
              inherit pp;
              extraPackages = [
                (depsWith pp).spherical-harmonics-encoding-jax
                (depsWith pp).volume-rendering-jax
              ];
            }))
        ];
        # REF:
        #   <https://github.com/google/jax/issues/5723#issuecomment-1339655621>
        XLA_FLAGS = with builtins; let
          nvidiaDriverVersion =
            head (match ".*Module  ([0-9\\.]+)  .*" (readFile /proc/driver/nvidia/version));
          nvidiaDriverVersionMajor = lib.toInt (head (splitVersion nvidiaDriverVersion));
        in lib.optionalString
          (nvidiaDriverVersionMajor <= 470)
          "--xla_gpu_force_compilation_parallelism=1";
        shellHook = ''
          source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.nixGLIntel})
          source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.auto.nixGLNvidia}*)
          [[ "$-" == *i* ]] && exec "$SHELL"
        '';
      };

      cpuDevShell = cpuPkgs.mkShell {
        name = "cpu";
        buildInputs = with cpuPkgs; [
          colmap
          (python3.withPackages (pp: mkPythonDeps {
              inherit pp;
              extraPackages = [];
            }))
        ];
        shellHook = ''
          [[ "$-" == *i* ]] && exec "$SHELL"
        '';
      };

      shjax = cudaPkgs.${py}.pkgs.callPackage ./deps/spherical-harmonics-encoding-jax {};
      volrendjax = cudaPkgs.${py}.pkgs.callPackage ./deps/volume-rendering-jax {};
    };
  });
}
