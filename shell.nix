{ pkgs ? import <nixpkgs> {} }:

let
  py = pkgs.python3;
  # dearpygui is only needed for funautoscripter_gui.py. If this nixpkgs revision
  # does not have it, the shellHook installs it into a local folder (.dpg) without
  # touching the system.
  hasDpg = py.pkgs ? dearpygui;
  python = py.withPackages (ps:
    [
      ps.numpy
      ps.scipy
      ps.opencv4      # CPU build is enough — classical optical flow only
      ps.matplotlib   # only needed for --plot
    ] ++ pkgs.lib.optional hasDpg ps.dearpygui);

  # Runtime libraries that dearpygui loads dynamically (OpenGL and X11).
  libs = with pkgs; [
    libGL
    libGLU
    xorg.libX11
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi
    xorg.libXext
    stdenv.cc.cc.lib
    zlib
  ];
in
pkgs.mkShell {
  packages = [ pkgs.ffmpeg python pkgs.clinfo pkgs.ocl-icd ]
    ++ pkgs.lib.optional (!hasDpg) py.pkgs.pip;

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}:${pkgs.ocl-icd}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
    # OpenCL (--opencl): the loader (ocl-icd) looks for GPU drivers here. The drivers
    # themselves (intel-compute-runtime, rocmPackages.clr.icd, or the NVIDIA driver)
    # come from the NixOS system config (hardware.graphics.extraPackages), not this
    # shell. To check what is available:
    #   python funautoscripter.py --accel-test        or        clinfo -l
    export OCL_ICD_VENDORS="''${OCL_ICD_VENDORS:-/run/opengl-driver/etc/OpenCL/vendors}"
  '' + pkgs.lib.optionalString (!hasDpg) ''
    if ! python -c "import dearpygui" 2>/dev/null; then
      echo "dearpygui not found in nixpkgs, installing locally into .dpg ..."
      pip install --quiet --target "$PWD/.dpg" dearpygui
    fi
    export PYTHONPATH="$PWD/.dpg:$PYTHONPATH"
  '';
}
