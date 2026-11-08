# Corpus checks

## 1. Held-out item leakage: 0 found


## 2. Specific details (phone/time/address): 0 found


## 3. KB consistency: 13,411 item+destination sentences checked

Hand-written seed documents: 0 conflicts (must be 0).


Generated documents: 213 flags (informational: generated text is KB-consistent by construction; flags are sentences about part of an item, e.g. a box's plastic liner or a jar's leftover food).

- train-00054 [generator]: 'Plastic takeout containers: Empty out leftover food into the green cart.' -> item takeout_container is recycling, sentence says organics
- train-00084 [generator]: 'Boxboard boxes: Take out the inner plastic bag; it goes in the garbage.' -> item cereal_box is recycling, sentence says garbage
- train-00093 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00093 [generator]: 'A: Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00105 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00145 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00151 [generator]: 'One more thing: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00154 [generator]: 'Note: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00184 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00187 [generator]: 'Cereal boxes: Pull out the plastic liner first and throw it in the garbage.' -> item cereal_box is recycling, sentence says garbage
- train-00257 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00316 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00346 [generator]: 'Barbecue grills: Remove the propane tank first and take it to a hazardous waste drop-off.' -> item bbq is bulky, sentence says hazardous
- train-00395 [generator]: 'Keep in mind: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00400 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00416 [generator]: 'One more thing: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00417 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00555 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00590 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00636 [generator]: 'Butter tubs: Scrape any leftover food into the green organics cart.' -> item margarine_tub is recycling, sentence says organics
- train-00737 [generator]: 'Aluminum trays: Empty out leftover food into the green cart.' -> item foil_tray is recycling, sentence says organics
- train-00755 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00775 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00775 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00785 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00800 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00800 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00822 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00824 [generator]: 'Yogurt containers: Empty out leftover food into the green cart.' -> item yogurt_container is recycling, sentence says organics
- train-00824 [generator]: 'Plastic nut butter jars: Scrape any leftover food into the green organics cart.' -> item peanut_butter_jar is recycling, sentence says organics
- train-00900 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00924 [generator]: 'Barbecue grills: Remove the propane tank first and take it to a hazardous waste drop-off.' -> item bbq is bulky, sentence says hazardous
- train-00926 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00951 [generator]: 'Margarine tubs: Empty out leftover food into the green cart.' -> item margarine_tub is recycling, sentence says organics
- train-00982 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00996 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-01046 [generator]: 'Grills: Take off the propane tank; it goes to hazardous waste.' -> item bbq is bulky, sentence says hazardous
- train-01058 [generator]: 'One more thing: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01168 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-01187 [generator]: 'Foam takeout containers: Scrape any leftover food into the green organics cart.' -> item foam_takeout is garbage, sentence says organics

## 4. Duplicate documents: 0


## 5. Sentence overlap with training split (informational)

| split | sentences | also in train | share |
|---|---:|---:|---:|
| val | 4,983 | 3,610 | 72.4% |
| test | 5,188 | 3,893 | 75.0% |
