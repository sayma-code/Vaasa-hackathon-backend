"""Extract the embedded photos and camera poses from a Matterport E57 export.

Writes scan_images/sweepNN_faceK.jpg and scan_images/poses.json.
Usage: python extract_e57.py [cloud_0.e57] [scan_images]
"""
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PAGE = 1024       # E57 physical page size
PAYLOAD = PAGE - 4  # each page ends with a 4-byte CRC
BLOB_HEADER = 16
NS = {"e": "http://www.astm.org/COMMIT/E57/2010-e57-v1.0"}


def read_logical(f, offset, length):
    """Read `length` payload bytes starting at physical `offset`, skipping page CRCs."""
    out = bytearray()
    pos = offset
    while len(out) < length:
        page_start = pos - pos % PAGE
        n = min(length - len(out), page_start + PAYLOAD - pos)
        f.seek(pos)
        out += f.read(n)
        pos = page_start + PAGE
    return bytes(out)


def skip_logical(offset, length):
    """Physical offset reached after advancing `length` payload bytes from `offset`."""
    pos = offset
    while length > 0:
        page_start = pos - pos % PAGE
        n = min(length, page_start + PAYLOAD - pos)
        length -= n
        pos = pos + n if length == 0 and pos + n < page_start + PAYLOAD else page_start + PAGE
    return pos


def num(node, path, default=0.0):
    el = node.find(path, NS)
    return float(el.text) if el is not None and el.text else default


def pose_of(node):
    return {
        "rotation_wxyz": [num(node, f"e:pose/e:rotation/e:{k}", 1.0 if k == "w" else 0.0) for k in "wxyz"],
        "translation_xyz": [num(node, f"e:pose/e:translation/e:{k}") for k in "xyz"],
    }


def text(node, path):
    el = node.find(path, NS)
    return el.text if el is not None else None


def main(src="cloud_0.e57", dst="scan_images"):
    out_dir = Path(dst)
    out_dir.mkdir(exist_ok=True)
    with open(src, "rb") as f:
        sig, _, _, _, xml_off, xml_len, _ = struct.unpack("<8sIIQQQQ", f.read(48))
        assert sig == b"ASTM-E57", "not an E57 file"
        root = ET.fromstring(read_logical(f, xml_off, xml_len))

        sweeps = {}
        for i, s in enumerate(root.findall("e:data3D/e:vectorChild", NS)):
            sweeps[text(s, "e:guid")] = {
                "index": i,
                "name": " ".join(text(s, "e:name").split()),
                "points": int(s.find("e:points", NS).get("recordCount")),
                **pose_of(s),
            }

        images = []
        for img in root.findall("e:images2D/e:vectorChild", NS):
            rep = img.find("e:pinholeRepresentation", NS)
            blob = rep.find("e:jpegImage", NS)
            sweep = sweeps[text(img, "e:associatedData3DGuid")]
            face = int(text(img, "e:name").split()[-1])
            fname = f"sweep{sweep['index']:02d}_face{face}.jpg"

            data_off = skip_logical(int(blob.get("fileOffset")), BLOB_HEADER)
            jpeg = read_logical(f, data_off, int(blob.get("length")))
            assert jpeg[:2] == b"\xff\xd8", f"{fname}: blob is not a JPEG"
            (out_dir / fname).write_bytes(jpeg)

            pixel = num(rep, "e:pixelWidth")
            images.append({
                "file": fname,
                "sweep": sweep["index"],
                "face": face,
                "width": int(num(rep, "e:imageWidth")),
                "height": int(num(rep, "e:imageHeight")),
                # pinhole intrinsics in pixels
                "fx": num(rep, "e:focalLength") / pixel,
                "fy": num(rep, "e:focalLength") / num(rep, "e:pixelHeight"),
                "cx": num(rep, "e:principalPointX"),
                "cy": num(rep, "e:principalPointY"),
                # camera-to-world pose
                **pose_of(img),
            })
            print(fname, f"{len(jpeg) / 1e6:.1f} MB")

    convention = ("Poses are camera-to-world: p_cam = R(rotation)^T @ (p_world - translation). "
                  "Camera looks along -Z with +X right and +Y up, so "
                  "u = cx + fx * x / -z, v = cy - fy * y / -z.")
    meta = {"source": src, "convention": convention, "sweeps": sorted(sweeps.values(), key=lambda s: s["index"]), "images": images}
    (out_dir / "poses.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(images)} images from {len(sweeps)} sweeps -> {out_dir}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
