#!/usr/bin/env python


import subprocess
import shlex
import re
import platform
import tempfile
import os
import sys
import argparse
import time

DEFAULT_VIRTUAL_WIDTH = 1024
DEFAULT_VIRTUAL_HEIGHT = 768

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def generate_xorg_conf(devices, virtual_width=DEFAULT_VIRTUAL_WIDTH, virtual_height=DEFAULT_VIRTUAL_HEIGHT):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual {virtual_width} {virtual_height}
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(
            device_id=i,
            screen_id=i,
            virtual_width=virtual_width,
            virtual_height=virtual_height,
        ))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))

    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    return "\n".join(xorg_conf)

def pci_slot_to_bus_id(slot):
    parts = [int(x, 16) for x in re.split(r'[:\.]', slot)]
    if len(parts) == 3:
        bus, device, function = parts
        return "PCI:%s:%s:%s" % (bus, device, function)
    if len(parts) == 4:
        domain, bus, device, function = parts
        return "PCI:%s@%s:%s:%s" % (bus, domain, device, function)
    raise ValueError("unrecognized PCI slot format: %s" % slot)

def nvidia_bus_ids():
    devices = []
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation' \
                and r.get('Class', '') in ['VGA compatible controller', '3D controller']:
            devices.append(pci_slot_to_bus_id(r['Slot']))
    return devices

def xorg_command(display):
    return [
        "Xorg",
        "-noreset",
        "+extension", "GLX",
        "+extension", "RANDR",
        "+extension", "RENDER",
        ":%s" % display,
    ]

def display_is_ready(display):
    env = os.environ.copy()
    env["DISPLAY"] = ":%s" % display
    return subprocess.run(
        ["xdpyinfo"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

def wait_for_display(display, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if display_is_ready(display):
            return True
        time.sleep(0.1)
    return False

def resize_display(display, virtual_width, virtual_height):
    env = os.environ.copy()
    env["DISPLAY"] = ":%s" % display
    subprocess.check_call(["xrandr", "--fb", "%sx%s" % (virtual_width, virtual_height)], env=env)

def call_xorg(command, display, virtual_width=None, virtual_height=None):
    process = subprocess.Popen(command)
    try:
        if virtual_width and virtual_height:
            if wait_for_display(display, process):
                resize_display(display, virtual_width, virtual_height)
                print("Set virtual screen size to %sx%s" % (virtual_width, virtual_height), flush=True)
            else:
                print("Warning: X display :%s was not ready before resize timeout." % display, flush=True)
        return process.wait()
    except Exception:
        process.terminate()
        process.wait()
        raise
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()

def startx(display, gpu_index=0, bus_id=None, use_generated_config=None,
           virtual_width=DEFAULT_VIRTUAL_WIDTH, virtual_height=DEFAULT_VIRTUAL_HEIGHT):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    devices = nvidia_bus_ids()

    if not devices:
        raise Exception("no nvidia cards found")

    if bus_id is None:
        if gpu_index < 0 or gpu_index >= len(devices):
            raise Exception("gpu index %s is out of range; found devices: %s" % (gpu_index, devices))
        bus_id = devices[gpu_index]

    if use_generated_config is None:
        use_generated_config = os.getuid() == 0

    command = xorg_command(display)
    if use_generated_config:
        if os.getuid() != 0:
            raise Exception(
                "Xorg only accepts arbitrary -config files when started with real UID 0. "
                "Run without --use-generated-config to use the non-root -isolateDevice path."
            )

        fd, path = tempfile.mkstemp(prefix='embodiedbench-xorg-', suffix='.conf')
        try:
            with os.fdopen(fd, "w") as f:
                f.write(generate_xorg_conf(devices, virtual_width, virtual_height))
            command[-1:-1] = ["-config", path]
            print("Using generated Xorg config for NVIDIA devices: %s" % ", ".join(devices), flush=True)
            return subprocess.call(command)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    command[-1:-1] = ["-isolateDevice", bus_id]
    print("Using Xorg -isolateDevice %s" % bus_id, flush=True)
    return call_xorg(command, display, virtual_width, virtual_height)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Start an Xorg server for AI2-THOR/ALFRED rendering.")
    parser.add_argument("display", nargs="?", type=int, default=0, help="X display number, e.g. 1 for DISPLAY=:1")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVIDIA GPU index from lspci order for non-root mode")
    parser.add_argument("--bus-id", help='Explicit Xorg BusID, e.g. "PCI:33:0:0"')
    parser.add_argument("--width", type=int, default=DEFAULT_VIRTUAL_WIDTH, help="Virtual screen width")
    parser.add_argument("--height", type=int, default=DEFAULT_VIRTUAL_HEIGHT, help="Virtual screen height")
    parser.add_argument(
        "--use-generated-config",
        action="store_true",
        help="Use the legacy generated xorg.conf path. This requires real root/sudo.",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print("Starting X on DISPLAY=:%s" % args.display, flush=True)
    sys.exit(startx(
        args.display,
        gpu_index=args.gpu_index,
        bus_id=args.bus_id,
        use_generated_config=True if args.use_generated_config else None,
        virtual_width=args.width,
        virtual_height=args.height,
    ))
