{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    let
      system = "x86_64-linux";

      # import nixpkgs and poetry2nix functions
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      inherit (poetry2nix.legacyPackages.${system}) mkPoetryApplication mkPoetryPackages mkPoetryEnv;

      # specify information about the package
      projectDir = self;
      python = pkgs.python310;
      # fix missing dependencies of external packages
      overrides = pkgs.poetry2nix.overrides.withDefaults (self: super: {
        confection = super.confection.overridePythonAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ self.setuptools ];
        });
      });

      # generate development environment
      poetry-env = mkPoetryEnv {
        inherit projectDir python overrides;
        preferWheels = true;
        groups = [ "dev" "test" ];
      };
      
    in
      {
        devShells.${system}.default = pkgs.mkShell {
          buildInputs = [
            pkgs.poetry
            pkgs.nodePackages.pyright
            poetry-env
          ];
        };
      };
}
