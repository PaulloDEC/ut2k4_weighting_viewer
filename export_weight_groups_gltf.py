"""
UT2K4 Weight Group glTF Exporter
=================================
Blender 4.5 LTS compatible script.

USAGE:
  1. Import your UT2K4 model via the PSK/PSA importer add-on as normal.
  2. Select the mesh object you want to export.
  3. Open this script in Blender's Text Editor and press Run Script.
  4. A .glb file will be saved alongside your .blend file (or to EXPORT_PATH).

APPROACH:
  1. Export a clean geometry-only GLB from Blender (no weight data).
  2. Read the vertex positions back from the exported GLB to get the exact
     vertex order Blender used after triangulation/splitting.
  3. For each exported vertex, find the closest matching original vertex and
     look up its weights. This handles any reordering Blender applies.
  4. Inject the correctly-ordered weight arrays directly into the GLB binary
     as _WG_N SCALAR FLOAT accessors.
"""

import bpy
import os
import json
import struct
import array
import mathutils

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EXPORT_PATH = ""

# ---------------------------------------------------------------------------
# GLB READER - extract vertex positions from exported GLB
# ---------------------------------------------------------------------------

def read_glb_positions(glb_path):
    """
    Parse the GLB and return a list of (x,y,z) tuples in the exact vertex
    order Blender wrote them into the binary buffer.
    """
    with open(glb_path, 'rb') as f:
        data = f.read()

    dv = struct.unpack_from

    # JSON chunk
    json_len = struct.unpack_from('<I', data, 12)[0]
    json_bytes = data[20 : 20 + json_len]
    gltf = json.loads(json_bytes.decode('utf-8'))

    # BIN chunk
    bin_start = 20 + json_len
    while bin_start % 4 != 0:
        bin_start += 1
    bin_len = struct.unpack_from('<I', data, bin_start)[0]
    bin_data = data[bin_start + 8 : bin_start + 8 + bin_len]

    # Find POSITION accessor
    positions = []
    for mesh in gltf.get('meshes', []):
        for prim in mesh.get('primitives', []):
            pos_acc_idx = prim.get('attributes', {}).get('POSITION')
            if pos_acc_idx is None:
                continue

            acc = gltf['accessors'][pos_acc_idx]
            bv  = gltf['bufferViews'][acc['bufferView']]

            byte_offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            count       = acc['count']
            byte_stride = bv.get('byteStride', 12)  # VEC3 float32 = 12 bytes default

            for i in range(count):
                off = byte_offset + i * byte_stride
                x, y, z = struct.unpack_from('<fff', bin_data, off)
                positions.append((x, y, z))

            return positions  # return first primitive's positions

    return positions


# ---------------------------------------------------------------------------
# GLB INJECTOR - append weight accessors into an existing GLB
# ---------------------------------------------------------------------------

def inject_weights_into_glb(glb_path, weights_per_group, group_names, num_exported_verts):
    with open(glb_path, 'rb') as f:
        data = f.read()

    # Parse JSON chunk
    json_len = struct.unpack_from('<I', data, 12)[0]
    json_bytes = data[20 : 20 + json_len]
    gltf = json.loads(json_bytes.decode('utf-8'))

    # Parse BIN chunk
    bin_start = 20 + json_len
    while bin_start % 4 != 0:
        bin_start += 1
    bin_data_len, bin_type = struct.unpack_from('<II', data, bin_start)
    existing_bin = bytearray(data[bin_start + 8 : bin_start + 8 + bin_data_len])

    buffer_views = gltf.setdefault('bufferViews', [])
    accessors    = gltf.setdefault('accessors',   [])

    if 'buffers' not in gltf or len(gltf['buffers']) == 0:
        gltf['buffers'] = [{'byteLength': len(existing_bin)}]

    wg_accessor_indices = []
    new_bin = bytearray(existing_bin)

    for gi, group_weights in enumerate(weights_per_group):
        # Pad to 4-byte alignment
        while len(new_bin) % 4 != 0:
            new_bin.append(0)

        byte_offset = len(new_bin)
        float_bytes = array.array('f', group_weights).tobytes()
        new_bin.extend(float_bytes)

        bv_idx = len(buffer_views)
        buffer_views.append({
            'buffer': 0,
            'byteOffset': byte_offset,
            'byteLength': len(float_bytes),
            'target': 34962
        })

        acc_idx = len(accessors)
        accessors.append({
            'bufferView': bv_idx,
            'byteOffset': 0,
            'componentType': 5126,  # FLOAT
            'count': num_exported_verts,
            'type': 'SCALAR',
            'normalized': False
        })
        wg_accessor_indices.append(acc_idx)

    # Inject as mesh primitive attributes (_WG_N = custom per glTF spec)
    for mesh in gltf.get('meshes', []):
        extras = mesh.setdefault('extras', {})
        extras['wg_count'] = len(group_names)
        extras['wg_names'] = ','.join(group_names)
        for prim in mesh.get('primitives', []):
            attrs = prim.setdefault('attributes', {})
            for gi, acc_idx in enumerate(wg_accessor_indices):
                attrs[f'_WG_{gi}'] = acc_idx

    gltf['buffers'][0]['byteLength'] = len(new_bin)

    # Re-serialise JSON (pad with spaces to 4-byte boundary)
    new_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(new_json) % 4 != 0:
        new_json += b' '

    # Re-serialise BIN (pad with zeros to 4-byte boundary)
    new_bin_bytes = bytes(new_bin)
    while len(new_bin_bytes) % 4 != 0:
        new_bin_bytes += b'\x00'

    json_chunk = struct.pack('<II', len(new_json),      0x4E4F534A) + new_json
    bin_chunk  = struct.pack('<II', len(new_bin_bytes), 0x004E4942) + new_bin_bytes
    header     = struct.pack('<III', 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))

    with open(glb_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    print(f"  Injected {len(wg_accessor_indices)} weight accessors")
    print(f"  Buffer: {len(existing_bin)} -> {len(new_bin_bytes)} bytes")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def export_weight_groups():
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        raise RuntimeError("Please select a mesh object before running this script.")

    vertex_groups = obj.vertex_groups
    if not vertex_groups:
        raise RuntimeError(f"Object '{obj.name}' has no vertex groups.")

    print(f"\n=== UT2K4 Weight Group Exporter ===")
    print(f"Mesh   : {obj.name}")
    print(f"Groups : {len(vertex_groups)}")
    print(f"Verts  : {len(obj.data.vertices)}")

    # Duplicate for export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate(linked=False)
    export_obj = bpy.context.active_object
    export_mesh = export_obj.data

    try:
        _do_export(obj, export_obj, export_mesh, vertex_groups)
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        export_obj.select_set(True)
        bpy.ops.object.delete()
        print("Temporary duplicate removed.")


def _do_export(original_obj, export_obj, export_mesh, vertex_groups):

    # Clean up duplicate
    for a in list(export_mesh.color_attributes):
        try:
            export_mesh.color_attributes.remove(a)
        except RuntimeError:
            pass
    export_obj.data.materials.clear()

    # Determine output path
    if EXPORT_PATH:
        out_path = EXPORT_PATH
    else:
        blend_path = bpy.data.filepath
        if not blend_path:
            out_path = os.path.join(bpy.app.tempdir, f"{original_obj.name}_weights.glb")
        else:
            out_path = os.path.join(os.path.dirname(blend_path), f"{original_obj.name}_weights.glb")

    # Step 1: Export geometry-only GLB
    bpy.ops.object.select_all(action='DESELECT')
    export_obj.select_set(True)
    bpy.context.view_layer.objects.active = export_obj

    print(f"\nStep 1: Exporting geometry to: {out_path}")
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        use_selection=True,
        export_format='GLB',
        export_apply=True,
        export_normals=True,
        export_attributes=False,
        export_vertex_color='NONE',
        export_materials='NONE',
        export_animations=False,
        export_extras=False,
    )

    # Step 2: Read back the exported vertex positions to get Blender's vertex order
    print("\nStep 2: Reading exported vertex order from GLB...")
    exported_positions = read_glb_positions(out_path)
    num_exported_verts = len(exported_positions)
    print(f"  Exported vertex count: {num_exported_verts}")

    # Step 3: Build a spatial lookup from original mesh vertices
    # Map each exported vertex position back to the original vertex index
    print("\nStep 3: Matching exported vertices to original mesh...")

    # Build KD-tree from original mesh vertices for fast nearest-neighbour lookup
    original_mesh = original_obj.data
    kd = mathutils.kdtree.KDTree(len(original_mesh.vertices))
    for v in original_mesh.vertices:
        # glTF uses Y-up; Blender uses Z-up. The exporter applies export_yup by default.
        # Transform: glTF(x,y,z) = Blender(x, z, -y)
        # So to go back: Blender(x,y,z) -> glTF(x, z, -y)
        kd.insert((v.co.x, v.co.z, -v.co.y), v.index)
    kd.balance()

    # Build weight lookup from original vertices
    blender_idx_to_export_idx = {vg.index: i for i, vg in enumerate(vertex_groups)}
    orig_weights = {}  # orig_vertex_index -> {group_export_idx: weight}
    for v in original_mesh.vertices:
        orig_weights[v.index] = {}
        for g in v.groups:
            ei = blender_idx_to_export_idx.get(g.group)
            if ei is not None:
                orig_weights[v.index][ei] = g.weight

    # For each exported vertex, find matching original vertex
    group_names  = [vg.name for vg in vertex_groups]
    num_groups   = len(group_names)

    # weights_per_group[gi][exported_vi] = float
    weights_per_group = [[0.0] * num_exported_verts for _ in range(num_groups)]

    unmatched = 0
    for exp_vi, (ex, ey, ez) in enumerate(exported_positions):
        co, orig_vi, dist = kd.find((ex, ey, ez))
        if dist > 0.001:
            unmatched += 1
        ow = orig_weights.get(orig_vi, {})
        for gi in range(num_groups):
            weights_per_group[gi][exp_vi] = ow.get(gi, 0.0)

    if unmatched > 0:
        print(f"  WARNING: {unmatched} vertices had no close match (dist > 0.001)")
    else:
        print(f"  All {num_exported_verts} vertices matched successfully")

    # Step 4: Inject weights into GLB
    print("\nStep 4: Injecting weight data into GLB...")
    inject_weights_into_glb(out_path, weights_per_group, group_names, num_exported_verts)

    print(f"\n✓ Export complete: {out_path}")
    print(f"  {num_groups} weight groups")
    print(f"  Groups: {group_names}")


if __name__ == "__main__":
    export_weight_groups()
