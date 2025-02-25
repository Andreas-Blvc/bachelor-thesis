from benchmark import config
from roads import load_road

entries_latex_code = ""
for road in config.Road:
    road_name = (" ".join([v.capitalize() for v in
            road.value
                .replace('./data/', '')
                .replace('.pkl', '')
                .replace('_1', '')
                .split('_')
            ]))
    road_obj = load_road('.' + road.value)
    entries_latex_code += "\multirow{" + str(len(road_obj.segments)) + "}{*}{" + road_name + "}"
    for i, s in enumerate(road_obj.segments):
        entries_latex_code += (
            f"&{i+1}"
            f"&{s.length:.1f}"
            f"&{s.get_curvature_at(0):.3f}"
            f"&[{s.n_min(0):.1f},{s.n_max(0):.1f}]"
            f"&[{s.n_min(s.length):.1f},{s.n_max(s.length):.1f}]"
            f"\\\\\n"
        )
    entries_latex_code += "\\midrule\n"
print(entries_latex_code)