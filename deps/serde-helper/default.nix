{ version, stdenvNoCC }: stdenvNoCC.mkDerivation {
  pname = "serde-helper";
  inherit version;
  src = ./serde.h;

  dontUnpack = true;

  installPhase = ''
    install -Dvm644 $src $out/include/serde-helper/serde.h
  '';
}
