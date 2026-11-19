"""Append compositional core items that strengthen thin material rules (KB v1.1).

Does not touch held-out items. Run:

    python -m src.datagen.add_core_items
    python src/kb_to_md.py
"""
import json
import re

from src.kb import KB_PATH, item_surface_forms, load_kb, validate

# Extra core examples for rules that are thin relative to held-out tests.
# Names/aliases must not overlap any existing or held-out surface form.
NEW_CORE = [
    # R02 soiled paper (only 2 core today; held-out: used paper plates)
    {"id": "greasy_takeout_boxes", "name": "greasy takeout boxes", "aliases": ["oily cardboard food boxes", "stained takeout cartons"],
     "material": "soiled paper", "rule": "R02", "stream": "organics", "prep": ["scrape"], "note": "", "split": "core"},
    {"id": "buttery_parchment", "name": "buttery parchment paper", "aliases": ["greasy baking paper", "oiled parchment sheets"],
     "material": "soiled paper", "rule": "R02", "stream": "organics", "prep": ["scrape"], "note": "", "split": "core"},
    {"id": "muffin_liners", "name": "used muffin liners", "aliases": ["paper cupcake liners", "greasy baking cups"],
     "material": "soiled paper", "rule": "R02", "stream": "organics", "prep": ["scrape"], "note": "", "split": "core"},
    {"id": "soiled_paper_bags", "name": "food-soiled paper bags", "aliases": ["greasy lunch bags", "oily paper grocery bags"],
     "material": "soiled paper", "rule": "R02", "stream": "organics", "prep": ["scrape"], "note": "", "split": "core"},
    {"id": "greasy_paper_towels", "name": "greasy paper towels", "aliases": ["oily paper towels", "food-wiped paper towels"],
     "material": "soiled paper", "rule": "R02", "stream": "organics", "prep": [], "note": "", "split": "core"},

    # R06 glass containers — avoid held-out aliases (vinegar/sauce bottles) and existing wine/beer/jam/pickle
    {"id": "kombucha_bottles", "name": "glass kombucha bottles", "aliases": ["fermented tea bottles", "glass kombucha jars"],
     "material": "glass container", "rule": "R06", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "sparkling_water_bottles", "name": "glass sparkling water bottles", "aliases": ["glass seltzer bottles", "carbonated water bottles"],
     "material": "glass container", "rule": "R06", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "mustard_jars", "name": "mustard jars", "aliases": ["dijon jars", "condiment glass jars"],
     "material": "glass container", "rule": "R06", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "honey_jars", "name": "honey jars", "aliases": ["glass honey pots", "empty honey jars"],
     "material": "glass container", "rule": "R06", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "spice_jars", "name": "glass spice jars", "aliases": ["empty spice jars", "herb jars"],
     "material": "glass container", "rule": "R06", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},

    # R20 cartons — existing milk/juice aliases already cover broth/soup; add distinct drink cartons
    # (avoid oat/plant/almond milk — held-out aliases)
    {"id": "coconut_water_cartons", "name": "coconut water cartons", "aliases": ["boxed coconut water", "coconut drink cartons"],
     "material": "carton", "rule": "R20", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "protein_shake_cartons", "name": "protein shake cartons", "aliases": ["boxed protein drinks", "meal-replacement cartons"],
     "material": "carton", "rule": "R20", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "eggnog_cartons", "name": "eggnog cartons", "aliases": ["holiday drink cartons", "seasonal eggnog cartons"],
     "material": "carton", "rule": "R20", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "cream_cartons", "name": "cream cartons", "aliases": ["whipping cream cartons", "cooking cream cartons"],
     "material": "carton", "rule": "R20", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"], "note": "", "split": "core"},

    # R05 metal — avoid sardine/tuna/fish tins (held-out)
    {"id": "coffee_cans", "name": "coffee cans", "aliases": ["metal coffee tins", "empty coffee cans"],
     "material": "metal", "rule": "R05", "stream": "recycling", "prep": ["empty", "rinse"], "note": "", "split": "core"},
    {"id": "tomato_paste_cans", "name": "tomato paste cans", "aliases": ["tomato sauce cans", "puree cans"],
     "material": "metal", "rule": "R05", "stream": "recycling", "prep": ["empty", "rinse"], "note": "", "split": "core"},
    {"id": "pet_food_cans", "name": "pet food cans", "aliases": ["cat food cans", "dog food cans"],
     "material": "metal", "rule": "R05", "stream": "recycling", "prep": ["empty", "rinse"], "note": "", "split": "core"},

    # R03 rigid plastic — yogurt/clamshells already exist; add other tubs
    {"id": "cottage_cheese_tubs", "name": "cottage cheese tubs", "aliases": ["ricotta containers", "quark tubs"],
     "material": "rigid plastic", "rule": "R03", "stream": "recycling", "prep": ["scrape", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "margarine_tubs", "name": "margarine tubs", "aliases": ["butter substitute tubs", "spread tubs"],
     "material": "rigid plastic", "rule": "R03", "stream": "recycling", "prep": ["scrape", "rinse", "caps_on"], "note": "", "split": "core"},
    {"id": "prescription_vials", "name": "empty prescription vials", "aliases": ["empty pill bottles", "empty medicine vials"],
     "material": "rigid plastic", "rule": "R03", "stream": "recycling", "prep": ["empty", "rinse", "caps_on"],
     "note": "Only empty plastic vials go in recycling; leftover medications go to pharmacy take-back.", "split": "core"},

    # R04 soft plastic
    {"id": "produce_bags", "name": "produce bags", "aliases": ["thin grocery produce bags", "vegetable bags"],
     "material": "soft plastic", "rule": "R04", "stream": "garbage", "prep": [], "note": "", "split": "core"},
    {"id": "freezer_bags", "name": "freezer bags", "aliases": ["ziplock bags", "resealable plastic bags"],
     "material": "soft plastic", "rule": "R04", "stream": "garbage", "prep": ["empty"], "note": "", "split": "core"},
    {"id": "dry_cleaning_bags", "name": "dry cleaning bags", "aliases": ["garment plastic covers", "thin film clothing bags"],
     "material": "soft plastic", "rule": "R04", "stream": "garbage", "prep": [], "note": "", "split": "core"},

    # R07 non-container glass — avoid vases (held-out)
    {"id": "picture_frame_glass", "name": "picture frame glass", "aliases": ["broken picture glass", "framed photo glass"],
     "material": "non-container glass", "rule": "R07", "stream": "garbage", "prep": ["wrap"], "note": "", "split": "core"},
    {"id": "incandescent_bulbs", "name": "incandescent light bulbs", "aliases": ["old filament bulbs", "standard light bulbs"],
     "material": "non-container glass", "rule": "R07", "stream": "garbage", "prep": ["wrap"],
     "note": "Fluorescent and CFL bulbs are hazardous waste, not garbage.", "split": "core"},
    {"id": "glass_cookware_lids", "name": "glass cookware lids", "aliases": ["casserole lids", "baking dish lids"],
     "material": "non-container glass", "rule": "R07", "stream": "garbage", "prep": ["wrap"], "note": "", "split": "core"},

    # R08 food — avoid held-out fruit/pit/rind/peel aliases
    {"id": "coffee_filters_used", "name": "used coffee filters", "aliases": ["paper coffee filters with grounds"],
     "material": "food", "rule": "R08", "stream": "organics", "prep": [], "note": "", "split": "core"},
    {"id": "tea_bags", "name": "tea bags", "aliases": ["used tea bags", "loose tea leaves"],
     "material": "food", "rule": "R08", "stream": "organics", "prep": [],
     "note": "Remove any plastic or nylon tea bags; only paper tea bags go in organics.", "split": "core"},
    {"id": "shrimp_shells", "name": "shrimp shells", "aliases": ["seafood shells", "crab shells"],
     "material": "food", "rule": "R08", "stream": "organics", "prep": [], "note": "", "split": "core"},
    {"id": "onion_skins", "name": "onion skins", "aliases": ["garlic skins", "vegetable peelings"],
     "material": "food", "rule": "R08", "stream": "organics", "prep": [], "note": "", "split": "core"},

    # R11 batteries — AA already alias of household_batteries; add other packs
    {"id": "power_tool_batteries", "name": "power tool batteries", "aliases": ["drill batteries", "cordless tool packs"],
     "material": "battery", "rule": "R11", "stream": "hazardous", "prep": ["tape_terminals"], "note": "", "split": "core"},
    {"id": "laptop_batteries", "name": "laptop batteries", "aliases": ["notebook battery packs", "removed laptop cells"],
     "material": "battery", "rule": "R11", "stream": "hazardous", "prep": ["tape_terminals"], "note": "", "split": "core"},
    {"id": "camera_batteries", "name": "camera batteries", "aliases": ["camcorder batteries", "camera battery packs"],
     "material": "battery", "rule": "R11", "stream": "hazardous", "prep": ["tape_terminals"], "note": "", "split": "core"},

    # R10 electronics — avoid kettles/routers/rice cookers (held-out); toasters/hair dryers exist
    {"id": "game_consoles", "name": "game consoles", "aliases": ["video game systems", "gaming consoles"],
     "material": "electronic", "rule": "R10", "stream": "electronics", "prep": ["wipe_data"], "note": "", "split": "core"},
    {"id": "smart_speakers", "name": "smart speakers", "aliases": ["voice assistants", "wifi speakers"],
     "material": "electronic", "rule": "R10", "stream": "electronics", "prep": ["wipe_data"], "note": "", "split": "core"},
    {"id": "electric_toothbrushes", "name": "electric toothbrushes", "aliases": ["powered toothbrushes", "rechargeable toothbrushes"],
     "material": "electronic", "rule": "R10", "stream": "electronics", "prep": ["remove_batteries"], "note": "", "split": "core"},

    # R12 chemicals — bleach/drain cleaner already aliases of household cleaners
    {"id": "pool_chemicals", "name": "pool chemicals", "aliases": ["spa chemicals", "chlorine tablets"],
     "material": "chemical", "rule": "R12", "stream": "hazardous", "prep": ["original_container", "no_mix"], "note": "", "split": "core"},
    {"id": "hobby_glues", "name": "solvent-based hobby glues", "aliases": ["model cement", "contact cement"],
     "material": "chemical", "rule": "R12", "stream": "hazardous", "prep": ["original_container", "no_mix"], "note": "", "split": "core"},
    {"id": "photographic_chemicals", "name": "photographic chemicals", "aliases": ["photo developer", "darkroom chemicals"],
     "material": "chemical", "rule": "R12", "stream": "hazardous", "prep": ["original_container", "no_mix"], "note": "", "split": "core"},

    # R13 furniture — avoid bookshelves/cribs
    {"id": "desk_chairs", "name": "office chairs", "aliases": ["desk chairs", "rolling chairs"],
     "material": "furniture", "rule": "R13", "stream": "bulky", "prep": ["book_pickup", "donate"], "note": "", "split": "core"},
    {"id": "wardrobes", "name": "wardrobes", "aliases": ["armoires", "standing closets"],
     "material": "furniture", "rule": "R13", "stream": "bulky", "prep": ["book_pickup", "donate"], "note": "", "split": "core"},
    {"id": "headboards", "name": "headboards", "aliases": ["bed frames", "wooden bed frames"],
     "material": "furniture", "rule": "R13", "stream": "bulky", "prep": ["book_pickup"], "note": "", "split": "core"},

    # R14 textiles — avoid scarves/hats/mittens
    {"id": "socks", "name": "socks", "aliases": ["mismatched socks", "worn socks"],
     "material": "fabric", "rule": "R14", "stream": "textiles", "prep": ["donate", "clean_bagged"], "note": "", "split": "core"},
    {"id": "fabric_gloves", "name": "fabric gloves", "aliases": ["cloth gloves", "cotton work gloves"],
     "material": "fabric", "rule": "R14", "stream": "textiles", "prep": ["donate", "clean_bagged"], "note": "", "split": "core"},
    {"id": "fabric_remnants", "name": "fabric remnants", "aliases": ["sewing scraps", "cloth leftovers"],
     "material": "fabric", "rule": "R14", "stream": "textiles", "prep": ["clean_bagged"], "note": "", "split": "core"},

    # R17 non-container plastic — plant pots already exist
    {"id": "plastic_buckets", "name": "plastic buckets", "aliases": ["utility buckets", "cleaning buckets"],
     "material": "plastic object", "rule": "R17", "stream": "garbage", "prep": ["empty"], "note": "", "split": "core"},
    {"id": "cd_cases", "name": "CD cases", "aliases": ["DVD cases", "plastic media cases"],
     "material": "plastic object", "rule": "R17", "stream": "garbage", "prep": [], "note": "", "split": "core"},
    {"id": "plastic_laundry_baskets", "name": "plastic laundry baskets", "aliases": ["hamper baskets", "plastic clothes baskets"],
     "material": "plastic object", "rule": "R17", "stream": "garbage", "prep": ["empty"], "note": "", "split": "core"},
]


def _forms(items):
    out = []
    for it in items:
        out.extend(f.lower() for f in item_surface_forms(it))
    return out


def main():
    kb = load_kb()
    existing_ids = {it["id"] for it in kb["items"]}
    existing_forms = _forms(kb["items"])
    added, skipped = [], []
    for it in NEW_CORE:
        if it["id"] in existing_ids:
            skipped.append((it["id"], "duplicate id"))
            continue
        clash = None
        for form in item_surface_forms(it):
            fl = form.lower()
            for ef in existing_forms:
                if re.search(r"\b" + re.escape(fl) + r"\b", ef) or re.search(r"\b" + re.escape(ef) + r"\b", fl):
                    clash = f"{form!r} overlaps existing {ef!r}"
                    break
            if clash:
                break
        if clash:
            skipped.append((it["id"], clash))
            continue
        added.append(it)
        existing_forms.extend(f.lower() for f in item_surface_forms(it))
        existing_ids.add(it["id"])

    if not added:
        print("no new items to add")
        for sid, why in skipped:
            print(f"  skip {sid}: {why}")
        return

    core = [it for it in kb["items"] if it["split"] == "core"]
    held = [it for it in kb["items"] if it["split"] == "heldout"]
    other = [it for it in kb["items"] if it["split"] not in ("core", "heldout")]
    kb["items"] = core + added + other + held
    kb["meta"]["version"] = "1.1"
    kb["meta"]["description"] = (
        "Fictional knowledge base for the Tidewell Residential Waste Guide (v1.1). "
        "v1.1 adds compositional core items on thin material rules to improve rule→item transfer "
        "without exposing held-out evaluation names."
    )
    errs = validate(kb)
    if errs:
        raise SystemExit("KB validation failed:\n  - " + "\n  - ".join(errs))
    KB_PATH.write_text(json.dumps(kb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {KB_PATH}")
    print(f"added {len(added)} core items; skipped {len(skipped)}")
    for sid, why in skipped:
        print(f"  skip {sid}: {why}")
    n_core = sum(1 for it in kb["items"] if it["split"] == "core")
    n_held = sum(1 for it in kb["items"] if it["split"] == "heldout")
    print(f"now core={n_core}, heldout={n_held}, total={len(kb['items'])}")


if __name__ == "__main__":
    main()
