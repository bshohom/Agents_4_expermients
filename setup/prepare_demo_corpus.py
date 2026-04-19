"""Create a tiny demo paper-card corpus and BOM file.

This is for testing repo plumbing without Elsevier or real full-text processing.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = [
        {
            "source_title": "Wire-Arc Additive Manufacturing of Low Alloy Steel Walls with Bead Geometry Control",
            "doi": "10.0000/demo-waam-steel-001",
            "process_family": "WAAM",
            "material_family": "low alloy steel",
            "material_system": ["ER70S-6", "mild steel substrate"],
            "feedstock_form": ["wire"],
            "substrate": ["mild steel plate"],
            "equipment": ["GMAW power source", "wire feeder", "robotic gantry", "shielding gas supply"],
            "consumables": ["ER70S-6 wire", "argon-CO2 shielding gas"],
            "controllable_parameters": ["current", "voltage", "travel speed", "wire feed speed", "interpass dwell"],
            "measurements_outputs": ["bead width", "wall height", "microhardness", "porosity", "macrography"],
            "heat_treatment": [],
            "microstructure_terms": ["ferrite", "bainite"],
            "bom_keywords": ["GMAW machine", "wire feeder", "steel substrate", "shielding gas", "robot"],
            "experiment_summary": "The study deposited straight WAAM walls using ER70S-6 wire while varying current, voltage, and travel speed to study bead geometry and hardness.",
            "unknowns": ["exact torch stand-off distance"],
            "feasibility_notes": "Feasible if a GMAW source, shielding gas, steel substrate, and travel system are available.",
        },
        {
            "source_title": "Laser Directed Energy Deposition of Ti-6Al-4V with Hatch Spacing and Laser Power Variation",
            "doi": "10.0000/demo-ded-ti64-002",
            "process_family": "DED",
            "material_family": "titanium alloy",
            "material_system": ["Ti-6Al-4V"],
            "feedstock_form": ["powder"],
            "substrate": ["Ti-6Al-4V plate"],
            "equipment": ["laser DED cell", "powder feeder", "argon glove enclosure", "pyrometer"],
            "consumables": ["Ti-6Al-4V powder", "argon gas"],
            "controllable_parameters": ["laser power", "scan speed", "powder feed rate", "hatch spacing"],
            "measurements_outputs": ["track width", "relative density", "microstructure", "hardness"],
            "heat_treatment": ["stress relief"],
            "microstructure_terms": ["martensitic alpha prime", "prior beta grain"],
            "bom_keywords": ["laser DED system", "powder feeder", "argon enclosure", "pyrometer", "Ti-6Al-4V powder"],
            "experiment_summary": "The study varied laser power and hatch spacing during DED of Ti-6Al-4V coupons and measured density and hardness.",
            "unknowns": ["exact layer cooling time"],
            "feasibility_notes": "Requires a laser DED cell and controlled atmosphere, so it is not feasible with only arc-welding equipment.",
        },
        {
            "source_title": "GTAW Bead-on-Plate Trials for 316L Stainless Steel with Travel Speed Variation",
            "doi": "10.0000/demo-gtaw-316l-003",
            "process_family": "GTAW",
            "material_family": "stainless steel",
            "material_system": ["316L stainless steel"],
            "feedstock_form": ["filler rod"],
            "substrate": ["316L plate"],
            "equipment": ["GTAW power source", "torch", "argon gas supply", "metallography saw"],
            "consumables": ["316L filler rod", "argon gas"],
            "controllable_parameters": ["current", "travel speed", "arc length"],
            "measurements_outputs": ["penetration depth", "bead width", "microstructure"],
            "heat_treatment": [],
            "microstructure_terms": ["austenite", "delta ferrite"],
            "bom_keywords": ["GTAW source", "torch", "argon", "316L plate"],
            "experiment_summary": "A bead-on-plate study on 316L varied travel speed during GTAW and examined resulting bead geometry and microstructure.",
            "unknowns": ["exact shielding gas flow rate"],
            "feasibility_notes": "Suitable when a TIG setup and metallography capability are available.",
        },
    ]

    bom = {
        "available_equipment": [
            "GMAW power source",
            "wire feeder",
            "steel substrate",
            "shielding gas supply",
            "robotic gantry",
            "microhardness tester",
        ],
        "available_consumables": [
            "ER70S-6 wire",
            "argon-CO2 shielding gas",
            "mild steel plate",
        ],
        "forbidden_equipment": [
            "laser DED cell",
            "powder feeder",
            "argon glove enclosure",
        ],
        "goal_constraints": {
            "target_material_family": "steel",
            "max_experiment_complexity": "medium",
            "prefer_existing_equipment": True,
        },
    }

    with (out_dir / "paper_cards_demo.jsonl").open("w", encoding="utf-8") as f:
        for row in cards:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "bom_demo.json").open("w", encoding="utf-8") as f:
        json.dump(bom, f, indent=2)

    print(f"Wrote demo corpus to: {out_dir}")


if __name__ == "__main__":
    main()
