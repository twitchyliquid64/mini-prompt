{ pkgs ? import <nixpkgs> {} }: with pkgs; mkShell {
  # ...
  nativeBuildInputs = [ pkg-config ];
  buildInputs = [ openssl ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [ openssl ];
}
