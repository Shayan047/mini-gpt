# Corpus checks

## 1. Held-out item leakage: 0 found


## 2. Specific details (phone/time/address): 0 found


## 3. KB consistency: 27,517 item+destination sentences checked

Hand-written seed documents: 0 conflicts (must be 0).


Generated documents: 296 flags (informational: generated text is KB-consistent by construction; flags are sentences about part of an item, e.g. a box's plastic liner or a jar's leftover food).

- train-00014 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00033 [generator]: 'Note: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00231 [generator]: 'Good to know: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00261 [generator]: 'Cracker boxes: Take out the inner plastic bag; it goes in the garbage.' -> item cereal_box is recycling, sentence says garbage
- train-00263 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00273 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00287 [generator]: 'Cereal boxes: Take out the inner plastic bag; it goes in the garbage.' -> item cereal_box is recycling, sentence says garbage
- train-00299 [generator]: 'One more thing: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00301 [generator]: 'Power tools: Pull out removable batteries and take them to a hazardous waste drop-off.' -> item power_tools is electronics, sentence says hazardous
- train-00308 [generator]: 'Margarine tubs: Empty out leftover food into the green cart.' -> item margarine_tub is recycling, sentence says organics
- train-00327 [generator]: 'Foam takeout containers: Scrape any leftover food into the green organics cart.' -> item foam_takeout is garbage, sentence says organics
- train-00466 [generator]: 'Sour cream tubs: Empty out leftover food into the green cart.' -> item yogurt_container is recycling, sentence says organics
- train-00493 [generator]: 'Tip: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00540 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00600 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-00640 [generator]: 'Note: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00695 [generator]: 'Game consoles: Pull out removable batteries and take them to a hazardous waste drop-off.' -> item game_console is electronics, sentence says hazardous
- train-00696 [generator]: 'Keep in mind: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00752 [generator]: 'Tip: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00763 [generator]: 'Plastic nut butter jars: Scrape any leftover food into the green organics cart.' -> item peanut_butter_jar is recycling, sentence says organics
- train-00809 [generator]: 'Plastic nut butter jars: Scrape any leftover food into the green organics cart.' -> item peanut_butter_jar is recycling, sentence says organics
- train-00838 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00846 [generator]: 'Plastic peanut butter jars: Scrape any leftover food into the green organics cart.' -> item peanut_butter_jar is recycling, sentence says organics
- train-00878 [generator]: 'One more thing: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00896 [generator]: 'Also, completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-00904 [generator]: 'Barbecue grills: Remove the propane tank first and take it to a hazardous waste drop-off.' -> item bbq is bulky, sentence says hazardous
- train-00936 [generator]: 'After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-00936 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-01023 [generator]: 'Grills: Take off the propane tank; it goes to hazardous waste.' -> item bbq is bulky, sentence says hazardous
- train-01030 [generator]: 'Tip: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01101 [generator]: 'Tip: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01116 [generator]: 'Butter tubs: Scrape any leftover food into the green organics cart.' -> item margarine_tub is recycling, sentence says organics
- train-01140 [generator]: 'Extra yard waste goes in paper bags next to your green cart.' -> item paper_bag is recycling, sentence says organics
- train-01236 [generator]: 'Note: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01248 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01352 [generator]: 'Answer: After the holidays, recycle plain wrapping paper and boxes, and put ribbons, bows and foil wrap in the garbage.' -> item wrapping_paper is recycling, sentence says garbage
- train-01396 [generator]: 'Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01453 [generator]: 'Tip: Completely empty, dried-out paint cans can go in the garbage.' -> item paint is hazardous, sentence says garbage
- train-01462 [generator]: 'Plastic nut butter jars: Empty out leftover food into the green cart.' -> item peanut_butter_jar is recycling, sentence says organics
- train-01482 [generator]: 'Cracker boxes: Take out the inner plastic bag; it goes in the garbage.' -> item cereal_box is recycling, sentence says garbage

## 4. Duplicate documents: 0


## 5. Sentence overlap with training split (informational)

| split | sentences | also in train | share |
|---|---:|---:|---:|
| val | 8,236 | 6,089 | 73.9% |
| test | 8,038 | 5,865 | 73.0% |
