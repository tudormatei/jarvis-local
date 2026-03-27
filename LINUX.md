# Getting it to run on Linux

## System Packages (via dnf)

- `gcc`, `python3-devel`, `gobject-introspection-devel`, `cairo-gobject-devel`, `pkg-config` — needed to build pygobject

## Conda Environment (pip installs)

- `pyqt5` + `qtpy` — Qt backend for pywebview
- `PyQtWebEngine` — required by pywebview for rendering HTML
- `evdev` — final working solution for push-to-talk on Linux/Wayland
- `pyinstaller` — for building the executable
- Downgraded `pywebview` from `6.1` → `4.4.1` — to fix the `MouseButtons.value` crash

## Code Changes

- Removed `import keyboard`, replaced push-to-talk `key_worker` with `evdev` implementation reading directly from `/dev/input/event4` (laptop keyboard) and `/dev/input/event6` (HyperX keyboard)
- Added `cuda` and `cuda.bindings.cydriver` to PyInstaller spec `collect_all` and `hiddenimports` to fix the NeMo/CUDA packaging issue

## Linux Spec File Changes vs Windows

- Forward slashes instead of backslashes in paths
- Icon changed from `.ico` to `.png`

# Building the application

## Step 1 — Copy the build to a permanent location

```sh
bashmkdir -p ~/.local/share/jarvis
cp -r dist/Jarvis/* ~/.local/share/jarvis/
```

## Step 2 — Copy the icon

```sh
bashmkdir -p ~/.local/share/icons
cp jarvis_ui/ui/assets/icon.png ~/.local/share/icons/jarvis.png
```

## Step 3 — Create the .desktop file

```sh
bashcat > ~/.local/share/applications/jarvis.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Jarvis
Comment=Jarvis AI Assistant
Exec=/home/tudor/.local/share/jarvis/Jarvis
Icon=/home/tudor/.local/share/icons/jarvis.png
Terminal=false
Categories=Utility;
StartupNotify=true
EOF
```

## Step 4 — Register it with GNOME

```sh
bashchmod +x ~/.local/share/applications/jarvis.desktop
update-desktop-database ~/.local/share/applications/
```
