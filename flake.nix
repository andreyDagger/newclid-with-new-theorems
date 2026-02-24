{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        boost = pkgs.boost;
        python = pkgs.python3;
        pythonEnv = python.withPackages (p: [
          # Here goes all the libraries that can't be managed by uv because of dynamic linking issues
          # or that you just want to be managed by nix for one reason or another
          p.numpy
          p.matplotlib
        ]);
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              uv
              python
              pythonEnv
              boost
            ];

            # scikit-build-core uses CMAKE_ARGS to pass options to CMake
            # - CMAKE_PREFIX_PATH points to boost.dev which contains the cmake config files
            # - USE_STATIC_LINK=OFF because Nix only provides shared Boost libraries
            CMAKE_ARGS = "-DCMAKE_PREFIX_PATH=${boost.dev} -DUSE_STATIC_LINK=OFF";

          # this runs when we do `nix develop .`

            shellHook = ''
              # Create a virtual environment if it doesn't exist
              if [ ! -d ".venv" ]; then
                uv venv .venv
              fi

              export UV_PYTHON_PREFERENCE="only-system";
              export UV_PYTHON=${python}
              source .venv/bin/activate

              uv sync --all-extras
              echo "Environment ready"
            '';
          };
      }
    );
}