{
  description = "Instant Neural Graphics Primitivs in JAX";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/22.11";
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
          ${py} = prev.${py}.override {
            packageOverrides = finalScope: prevScope: {
              jax = prevScope.jax.overridePythonAttrs (o: { doCheck = false; });
              chex = prevScope.chex.override {
                  jaxlib = prevScope.jaxlib-bin;
                };
              optax = prevScope.optax.override {
                  jaxlib = prevScope.jaxlib-bin;
                };
              flax = (prevScope.flax.override {
                  jaxlib = prevScope.jaxlib-bin;
                }).overridePythonAttrs (o: {
                    buildInputs = o.buildInputs ++ [ prevScope.pyyaml ];
                    # tensorflow is only a check input, let's assume nixpkgs@22.11 has a working
                    # flax pacakge and disable checks, so that tensorflow will not be built due to
                    # nix's lazy evaluation.
                    doCheck = false;
                  });
            };
          };
        };
      mkPythonDeps = { pp,  extraPackages }: with pp; [
          ipython
          pytorch-bin
          torchvision-bin
          tqdm
          icecream
          (depsWith pp).tyro
          pillow
          ipdb
          sympy
          colorama

          jaxlib-bin
          jax
          optax
          flax
        ] ++ extraPackages;
    in rec {
      default = cuda;
      cuda = let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [
            inputs.nixgl.overlays.default
            jaxOverlays
          ];
        };
      in pkgs.mkShell {  # impure
        name = "cuda";
        buildInputs = [
          (pkgs.${py}.withPackages (pp: mkPythonDeps {
              inherit pp;
              extraPackages = [];
            }))
        ];
        shellHook = ''
          source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.nixGLIntel})
          source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.auto.nixGLNvidia}*)
          [[ "$-" == *i* ]] && exec "$SHELL"
        '';
      };
      cpu = let
        pkgs = import nixpkgs {
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
      in pkgs.mkShell {
        name = "cpu";
        buildInputs = with pkgs; [
          (python3.withPackages (pp: mkPythonDeps {
              inherit pp;
              extraPackages = [];
            }))
        ];
        shellHook = ''
          [[ "$-" == *i* ]] && exec "$SHELL"
        '';
      };
    };
  });
}
