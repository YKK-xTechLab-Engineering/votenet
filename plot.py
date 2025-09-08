from pathlib import Path
import json
import numpy as np
import pyvista as pv
from matplotlib import cm
from votenet.utils.pc_util import convert_camera_corners_to_upright


p = pv.Plotter()


ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "demo_files"

scene_points = pv.read(DATA_DIR / "sunrgbd_results" / "000000_pc.ply")
# scene_points = pv.read(DATA_DIR / "input_pc_sunrgbd.ply")
# 
pred_map_cls_json = DATA_DIR / "sunrgbd_results" / "pred_map_cls.json"
type2class_path = DATA_DIR / "sunrgbd_results" / "type2class.txt"

# Predictions in pred_map_cls.json are in camera coordinates (SUNRGBD)
PRED_IN_CAMERA_COORDS = True


def build_class_color_mapping():
    """Build a deterministic color mapping for classes.

    Preference order:
    1) Use `type2class.txt` index ordering if available
    2) Fallback to classes found in the JSON, sorted
    """
    class_to_color = {}
    cmap = cm.get_cmap("tab20")

    classes = []
    if type2class_path.exists():
        # Lines are in format: "classname idx"
        text = type2class_path.read_text().strip()
        if text:
            tmp = {}
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        idx = int(parts[1])
                    except ValueError:
                        continue
                    tmp[idx] = name
            for idx in sorted(tmp.keys()):
                classes.append(tmp[idx])

    if not classes:
        try:
            preds = json.loads(pred_map_cls_json.read_text())
            classes = sorted({d.get("classname", "unknown") for d in preds})
        except Exception:
            classes = []

    for i, name in enumerate(classes):
        rgba = cmap(i % cmap.N)
        class_to_color[name] = tuple(float(c) for c in rgba[:3])

    return class_to_color


CLASS_TO_COLOR = build_class_color_mapping()


def color_for_class(name: str):
    if name not in CLASS_TO_COLOR:
        cmap = cm.get_cmap("tab20")
        idx = abs(hash(name)) % cmap.N
        rgba = cmap(idx)
        CLASS_TO_COLOR[name] = tuple(float(c) for c in rgba[:3])
    return CLASS_TO_COLOR[name]


def make_wireframe_from_bbox(points8: np.ndarray) -> pv.PolyData:
    """Create a PolyData containing the 12 edges of a box defined by 8 corners.

    points8: (8, 3) array of corner coordinates.
    Order is assumed consistent but robust to face ordering.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top face perimeter
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face perimeter
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    poly = pv.PolyData(points8)
    lines = np.hstack([[2, i, j] for (i, j) in edges]).astype(np.int64)
    poly.lines = lines
    return poly


def top_label_position(points8: np.ndarray) -> np.ndarray:
    """Return a point slightly above the top face center for label placement."""
    y = points8[:, 1]
    top_idx = np.argsort(y)[-4:]
    top_center = points8[top_idx].mean(axis=0)
    extent_y = float(y.max() - y.min()) if np.isfinite(y).all() else 0.0
    offset = max(0.02, 0.1 * extent_y) if extent_y > 0 else 0.05
    return top_center + np.array([0.0, offset, 0.0])

def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

if __name__ == "__main__":
    # Load predictions
    with pred_map_cls_json.open("r") as f:
        predictions = json.load(f)


    # Plot background point cloud
    p.add_mesh(scene_points, color="lightgray", opacity=0.4)


    # Draw wireframe boxes and labels
    for det in predictions:
        classname = det.get("classname", "unknown")
        bbox = np.array(det.get("bbox", []), dtype=float)
        score = float(det.get("score", 0.0))

        if bbox.shape != (8, 3):
            continue

        if PRED_IN_CAMERA_COORDS:
            bbox = convert_camera_corners_to_upright(bbox)

        color = color_for_class(classname)

        wire = make_wireframe_from_bbox(bbox)
        p.add_mesh(wire, color=color, line_width=2, render_lines_as_tubes=False)

        label_pos = top_label_position(bbox)
        p.add_point_labels(
            [label_pos],
            [f"{classname} {score:.2f}"],
            font_size=12,
            text_color=color,
            point_size=0,
            always_visible=True,
            shape=None,
        )


    p.show()